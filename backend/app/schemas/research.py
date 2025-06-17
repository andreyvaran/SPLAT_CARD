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


class NeironColorResult(MyBaseModel):
    color_module_1: str | None = None
    color_module_2: str | None = None
    result: float


class ResearchInnerResult(MyBaseModel):
    finish: bool = True
    error_message: str | None = None
    processed_image: str | None = None

    white_blood_cels: NeironColorResult | None = None
    read_blood_cels: NeironColorResult | None = None
    total_level_protein: NeironColorResult | None = None
    ph_level: NeironColorResult | None = None
    total_stiffness: NeironColorResult | None = None
