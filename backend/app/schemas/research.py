import datetime

from pydantic import UUID4

from app.schemas.base import MyBaseModel
from enum import Enum


class FileStatus(str, Enum):
    SavedRow = "SavedRow"
    Error = "Error"
    NotValid = "NotValid"
    Success = "Success"



class NeuronColorResult(MyBaseModel):
    color_module_1: str | None = None
    color_module_2: str | None = None
    result: float


class ResearchInnerResult(MyBaseModel):
    finish: bool = True
    error_message: str | None = None
    processed_image: str | None = None

    white_blood_cells: NeuronColorResult | None = None
    red_blood_cells: NeuronColorResult | None = None
    total_level_protein: NeuronColorResult | None = None
    ph_level: NeuronColorResult | None = None
    total_stiffness: NeuronColorResult | None = None



class ResearchData(MyBaseModel):
    id: UUID4
    result: ResearchInnerResult | None = None
    files: list[str]
    status: FileStatus
    create_date: datetime.datetime

