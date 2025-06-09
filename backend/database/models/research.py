import uuid


from app.schemas.research import ResearchData, FileStatus
from database.db_metadata import Base
from sqlalchemy import UUID, ForeignKey, String, JSON, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
import datetime


class ResearchORM(Base):
    __tablename__ = "research"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    result = mapped_column(JSON, nullable=True)
    files: Mapped[list[str] ] = mapped_column(ARRAY(String), default=list)
    create_date: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)
    status: Mapped[FileStatus | None]
    user_token_id: Mapped[uuid.UUID] = mapped_column(
        UUID,
        ForeignKey("user_token.id", ondelete="CASCADE"),
        index=True
    )
    user_token: Mapped["UserTokenORM"] = relationship(
        "UserTokenORM",
        back_populates="researches"
    )

    def get_schema(self) -> ResearchData:
        return ResearchData.model_validate(self)
