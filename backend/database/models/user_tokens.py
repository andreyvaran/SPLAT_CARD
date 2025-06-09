import hashlib
import uuid
from datetime import datetime

from app.schemas.user_token import UserTokenModel, UserTokenRequest
from database.db_metadata import Base
from database.models.mixin import IsActiveMixin
from sqlalchemy import  UUID, String
from sqlalchemy.orm import Mapped, mapped_column, relationship


class UserTokenORM(Base, IsActiveMixin):
    __tablename__ = "user_token"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    hex_token: Mapped[str] = mapped_column(
        String(32),
        default=lambda: hashlib.md5(datetime.now().isoformat().encode()).hexdigest()
    )

    researches: Mapped[list["ResearchORM"]] = relationship(
        "ResearchORM",
        back_populates="user_token",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    def get_schema(self) -> UserTokenModel:
        return UserTokenModel.model_validate(self)


    def get_token_schema(self) -> UserTokenRequest:
        return UserTokenRequest.model_validate(self)