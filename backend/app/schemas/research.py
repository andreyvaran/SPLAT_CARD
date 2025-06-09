import datetime

from pydantic import UUID4

from app.schemas.base import MyBaseModel
from enum import Enum

class FileStatus(str, Enum):
    SavedRow = "SavedRow"
    Error = "Error"
    NotValid = "NotValid"
    Success = "Success"


class ResearchData(MyBaseModel):
    id: UUID4
    result: dict | None = None
    files: list[str]
    status: FileStatus
    create_date: datetime.datetime

