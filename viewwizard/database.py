from collections.abc import AsyncGenerator, Collection, Generator, Sequence
import string
import asyncio
from dataclasses import dataclass, field
import io
import pathlib
import pickle
import random
import uuid

import aiohttp
import dotenv
import googleapiclient.discovery
import PIL.Image
import torch
import torchvision
import yarl

import viewwizard.schema as schema
import viewwizard.model as vmodel


@dataclass
class VideoData:
    video_data: schema.YouTubeVideoItemJSON
    video_data_id: uuid.UUID = field(init=False)
    thumbnail: yarl.URL | torch.Tensor = field(init=False)  # Has shape (3 x H x W)

    def __post_init__(self):
        self.thumbnail = yarl.URL(
            self.video_data["snippet"]["thumbnails"]["default"]["url"]
        )

        self.video_data_id = uuid.uuid4()

    async def image_tensor(
        self,
        session: aiohttp.ClientSession,
    ) -> torch.Tensor:
        if isinstance(self.thumbnail, yarl.URL):
            async with session.get(
                str(self.thumbnail)
                .replace("https: //", "https://")
                .replace("http: //", "http://")
            ) as response:
                response.raise_for_status()

                pil_image = PIL.Image.open(
                    io.BytesIO(await response.content.read())
                ).convert("RGB")

                self.thumbnail = torchvision.transforms.ToTensor()(pil_image)

        return self.thumbnail


def create_random_query(query_size: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=query_size))


def chunked[T](
    iterable: Sequence[T], chunk_size: int
) -> Generator[Sequence[T], None, None]:
    for index in range(0, len(iterable), chunk_size):
        yield iterable[index : index + chunk_size]


def create_client() -> googleapiclient.discovery.Resource:
    dotenv.load_dotenv()
    key = dotenv.dotenv_values()["YT_GOOGLE_API_KEY"]

    return googleapiclient.discovery.build(
        "youtube", "v3", developerKey=key, cache_discovery=False
    )


def sample_random_search_ids(
    client, query: str, video_duration: str = "medium"
) -> list[str]:
    response: schema.YouTubeSearchListResponseJSON = (
        client.search()
        .list(
            part="id",
            type="video",
            q=query,
            videoDuration=video_duration,
            maxResults=50,
        )
        .execute()
    )

    return [
        video["id"]["videoId"]
        for video in response["items"]
        if video["id"]["kind"] == "youtube#video"
    ]


class VideoDataset:
    dataset_id: uuid.UUID
    videos: dict[str, VideoData]
    video_ids: list[str]

    def __init__(
        self,
        *,
        videos: dict[str, VideoData] | None = None,
        video_ids: list[str] | None = None,
    ):
        self.videos = videos or {}
        self.video_ids = video_ids or []

        self.dataset_id = uuid.uuid4()

    def save_to_path(self, path: pathlib.Path):
        with path.open("wb") as file:
            pickle.dump(self, file)

    def save_to_path_compressed(self, path: pathlib.Path):
        with path.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_path(cls, path: pathlib.Path):
        with path.open("rb") as file:
            return pickle.load(file)

    async def pull_thumbnails(self, session: aiohttp.ClientSession):
        await asyncio.gather(
            *(video_data.image_tensor(session) for video_data in self.videos.values())
        )

    def add_ids(self, video_ids: Collection[str]):
        self.video_ids.extend(video_ids)

    def sync_id_video_data(self, youtube_client):
        unsynced_video_ids = [
            video_id for video_id in self.video_ids if video_id not in self.videos
        ]

        for id_chunk in chunked(unsynced_video_ids, 49):
            response: schema.YouTubeVideoListResponseJSON = (
                youtube_client.videos()
                .list(
                    part="snippet,statistics",
                    id=",".join(id_chunk),
                )
                .execute()
            )

            for video_data in response["items"]:
                self.videos[video_data["id"]] = VideoData(video_data)

    def shuffle_dataset(self):
        random.shuffle(self.video_ids)

    async def get_training_batches(
        self,
        batch_size: int,
    ) -> AsyncGenerator[vmodel.ThumbnailStatisticsTrainingBatch, None]:
        async with aiohttp.ClientSession() as aiohttp_session:
            for samples in chunked([*self.videos.items()], batch_size):
                try:
                    yield vmodel.ThumbnailStatisticsTrainingBatch(
                        video_ids=[video_id for video_id, _ in samples],
                        image=torch.stack(
                            [
                                await video_data.image_tensor(aiohttp_session)
                                for _, video_data in samples
                            ]
                        ),
                        view_count=torch.stack(
                            [
                                torch.tensor(
                                    int(
                                        video_data.video_data["statistics"]["viewCount"]
                                    )
                                )
                                for _, video_data in samples
                                if isinstance(video_data.thumbnail, torch.Tensor)
                            ]
                        ),
                    )

                except ValueError:
                    pass

    def split_dataset(
        self, split_ratio: float
    ) -> tuple["VideoDataset", "VideoDataset"]:
        split_index: int = int(len(self.video_ids) * split_ratio)

        return (
            VideoDataset(
                videos={
                    video_id: self.videos[video_id]
                    for video_id in self.video_ids[:split_index]
                },
                video_ids=self.video_ids[:split_index],
            ),
            VideoDataset(
                videos={
                    video_id: self.videos[video_id]
                    for video_id in self.video_ids[split_index:]
                },
                video_ids=self.video_ids[split_index:],
            ),
        )

    def extend_dataset(self, other_dataset: "VideoDataset"):
        self.videos.update(other_dataset.videos)
        self.video_ids.extend(other_dataset.video_ids)

    @classmethod
    def merge_datasets(cls, datasets: list["VideoDataset"]):
        merged_dataset = cls()

        for dataset in datasets:
            merged_dataset.extend_dataset(dataset)

        return merged_dataset
