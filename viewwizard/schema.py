from typing import TypedDict, NotRequired, List


# ---- THUMBNAILS -------------------------------------------------------------


class YouTubeThumbnailJSON(TypedDict):
    url: str
    width: NotRequired[int]
    height: NotRequired[int]


class YouTubeThumbnailSetJSON(TypedDict):
    default: YouTubeThumbnailJSON
    medium: NotRequired[YouTubeThumbnailJSON]
    high: NotRequired[YouTubeThumbnailJSON]
    standard: NotRequired[YouTubeThumbnailJSON]
    maxres: NotRequired[YouTubeThumbnailJSON]


# ---- LOCALIZED ---------------------------------------------------------------


class YouTubeLocalizedJSON(TypedDict):
    title: str
    description: str


# ---- SNIPPET -----------------------------------------------------------------


class YouTubeSnippetJSON(TypedDict):
    publishedAt: str
    channelId: str
    title: str
    description: str
    thumbnails: YouTubeThumbnailSetJSON
    channelTitle: str
    tags: NotRequired[List[str]]
    categoryId: str
    liveBroadcastContent: str
    defaultLanguage: NotRequired[str]
    localized: NotRequired[YouTubeLocalizedJSON]
    defaultAudioLanguage: NotRequired[str]


# ---- STATISTICS --------------------------------------------------------------


class YouTubeStatisticsJSON(TypedDict):
    viewCount: str
    likeCount: NotRequired[str]
    favoriteCount: str
    commentCount: NotRequired[str]


# ---- VIDEO ITEM --------------------------------------------------------------


class YouTubeVideoItemJSON(TypedDict):
    kind: str
    etag: str
    id: str
    snippet: YouTubeSnippetJSON
    statistics: YouTubeStatisticsJSON


# ---- PAGE INFO ---------------------------------------------------------------


class YouTubePageInfoJSON(TypedDict):
    totalResults: int
    resultsPerPage: int


# ---- TOP-LEVEL RESPONSE ------------------------------------------------------


class YouTubeVideoListResponseJSON(TypedDict):
    kind: str
    etag: str
    items: List[YouTubeVideoItemJSON]
    pageInfo: YouTubePageInfoJSON
