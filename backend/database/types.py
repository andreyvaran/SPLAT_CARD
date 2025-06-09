from uuid import UUID, uuid4

from typing import Annotated

from pydantic import Field

from sqlalchemy import String, Integer
from sqlalchemy.orm import mapped_column, declarative_base
from sqlalchemy.dialects.postgresql import UUID as SQLAlchemyUUID


Base = declarative_base()

str_20 = Annotated[
    str,
    Field(max_length=20),
    mapped_column(String(20), default=str, nullable=True),
]
str_64 = Annotated[
    str,
    Field(max_length=64),
    mapped_column(String(64), default=str, nullable=True),
]
str_256 = Annotated[str, Field(max_length=256), mapped_column(String(256), nullable=False)]
str_1024 = Annotated[str, Field(max_length=1024), mapped_column(String(1024), nullable=False)]

UUID_PK = Annotated[
    UUID,
    mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        primary_key=True,
        index=True,
        default=uuid4,
    ),
]

UUID_C = Annotated[
    UUID,
    mapped_column(SQLAlchemyUUID(as_uuid=True), default=uuid4, index=True),
]

int_pk = Annotated[int, mapped_column(Integer(), primary_key=True, index=True)]

int_c = Annotated[int, mapped_column(Integer, default=1)]
