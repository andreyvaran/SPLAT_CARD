from abc import abstractmethod
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import JSON

from sqlalchemy.inspection import inspect

from database.types import (
    UUID_PK,
    str_20,
    str_64,
    str_256,
    str_1024,
    int_pk,
    UUID_C,
    int_c,
)


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON,
        str_20: str_20,
        str_64: str_64,
        str_256: str_256,
        str_1024: str_1024,
        int_pk: int_pk,
        UUID_PK: UUID_PK,
        UUID_C: UUID_C,
        int_c: int_c,
    }

    @abstractmethod
    def get_schema(self) -> BaseModel:
        raise NotImplementedError

    @classmethod
    def get_related(self):
        return inspect(self).relationships.items()
