from abc import ABC
from typing import TypeVar
from uuid import UUID
from pydantic import BaseModel

from sqlalchemy import insert, select, update, delete, literal_column, RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from database.db_metadata import Base

ModelType = TypeVar("ModelType", bound=BaseModel)
ORMType = TypeVar("ORMType", bound=Base)
IDType = TypeVar("IDType", bound=int | UUID)
ListIDType = TypeVar("ListIDType", bound=list[int] | list[UUID])
ResultType = IDType | ListIDType


class AbstractRepository(ABC):
    pass


class SQLAlchemyRepository(AbstractRepository):
    model: ORMType = None

    def __init__(self, session: AsyncSession):
        self.session: AsyncSession = session

    async def add_one(self, data: dict) -> RowMapping | None:
        stmt = insert(self.model).values(**data).returning(literal_column("*"))
        res = await self.session.execute(stmt)
        return res.mappings().first()

    async def edit_one(self, id: int, data: dict) -> RowMapping | None:
        stmt = update(self.model).values(**data).filter_by(id=id).returning(literal_column("*"))
        res = await self.session.execute(stmt)
        return res.mappings().first()

    async def edit_by_filter(self, filters: dict, data: dict) -> ResultType | None:
        stmt_update = (
            update(self.model).values(**data).filter_by(**filters).returning(self.model.id)
        )
        result = await self.session.execute(stmt_update)
        ids = result.scalars().all()

        if len(ids) == 0:
            return None
        elif len(ids) == 1:
            return ids[0]
        else:
            return ids

    async def get_all(self) -> list[ModelType]:
        stmt = select(self.model)
        res = await self.session.execute(stmt)
        res = [row[0].get_schema() for row in res.all()]
        return res

    async def get_first(self) -> ORMType:
        stmt = select(self.model)
        res = await self.session.execute(stmt)
        res = res.scalar_one().get_schema()
        return res

    async def get_one(self, **filter_by) -> ORMType:
        stmt = select(self.model).filter_by(**filter_by)
        res = await self.session.execute(stmt)
        res = res.scalar()
        return res

    async def delete(self, **filter_by) -> None:
        stmt = delete(self.model).filter_by(**filter_by).returning(literal_column("*"))
        await self.session.execute(stmt)

    async def soft_delete(self, id: int) -> int:
        stmt = update(self.model).filter_by(id=id).values(is_active=False).returning(self.model)
        result = (await self.session.execute(stmt)).scalar()
        return result

    async def activate(self, id: int) -> int:
        stmt = update(self.model).filter_by(id=id).values(is_active=True)
        result = await self.session.execute(stmt)
        return result.rowcount

    async def get_by_id(self, id: int) -> ORMType:
        t = await self.session.get(self.model, id)
        return t

    async def get_count_by_filters(self, **filter_by) -> int:
        stmt = select(self.model).filter_by(**filter_by)
        res = await self.session.execute(stmt)
        return len(res.all())

    async def get_all_with_filters(self, **filter_by) -> list[ModelType]:
        stmt = select(self.model).filter_by(**filter_by)
        res = await self.session.execute(stmt)
        res = [row[0].get_schema() for row in res.all()]
        return res

    async def get_first_with_filters(self, **filter_by) -> ORMType:
        stmt = select(self.model).filter_by(**filter_by)
        res = await self.session.execute(stmt)
        res = res.first()
        return res

    async def get_attrs_with_filters(self, *attrs, **filter_by) -> list:
        # todo fix this
        stmt = select(*attrs).filter_by(**filter_by)
        res = await self.session.execute(stmt)
        res = [row[0] for row in res.all()]
        return res
