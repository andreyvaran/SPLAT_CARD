from enum import Enum


class Bucket(Enum):
    CONTENT: str = "content"
    AUTHORS: str = "content"
    ARTICLE_PREVIEW = "abstract"
    MAIN: str = "main"

