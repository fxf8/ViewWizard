from dataclasses import dataclass, field
import io

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
