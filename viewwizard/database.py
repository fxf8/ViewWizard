import asyncio
from dataclasses import dataclass, field
import io
import pathlib
import pickle

import aiohttp
import PIL.Image
import torch
import torchvision
import yarl

import viewwizard.schema as schema


@dataclass
class VideoData:
    video_data: schema.YouTubeVideoItemJSON
    thumbnail: yarl.URL | torch.Tensor = field(init=False)

    def __post_init__(self):
        self.thumbnail = yarl.URL(
            self.video_data["snippet"]["thumbnails"]["default"]["url"]
        )

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

                self.thumbnail = torch.from_numpy(
                    torchvision.transforms.ToTensor()(pil_image)
                )

        return self.thumbnail


@dataclass
class VideoDataset:
    videos: list[VideoData]

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

    async def pull_thumbnails(
        self, session: aiohttp.ClientSession
    ) -> asyncio.Future[list[torch.Tensor]]:
        return asyncio.gather(
            *(video_data.image_tensor(session) for video_data in self.videos)
        )
