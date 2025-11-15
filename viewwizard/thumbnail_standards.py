from dataclasses import dataclass


@dataclass(frozen=True)
class AspectRatioInfo:
    width: int
    height: int


YOUTUBE_THUMBNAIL_ASPECT_RATIO = AspectRatioInfo(width=1280, height=720)
