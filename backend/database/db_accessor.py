import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy import text
from config.db import DBSettings
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger("database")


class DatabaseAccessor:
    _db_settings = None
    DEFAULT_ACQUIRE_TIMEOUT: float = 1
    DEFAULT_REFRESH_DELAY: float = 1
    DEFAULT_REFRESH_TIMEOUT: float = 5

    def __init__(self, db_settings: DBSettings):
        # print(id(self), "Вызвали конструктор создания дб ацессора")
        self._db_settings = db_settings
        self._dsn = db_settings.dsn_async
        self._async_session_maker = None

    def set_settings(self, db_settings: DBSettings):
        self.__init__(db_settings)

    def _set_engine(self) -> None:
        if "sqlite" in self._dsn:
            self.engine = create_async_engine(
                self._dsn,
                connect_args={"check_same_thread": False},  # Важно для SQLite
                echo=False,
            )
        else:
            # print(self._db_settings.DB_POOL_SIZE)
            self.engine = create_async_engine(
                self._dsn,
                pool_size=20,  # Adjust pool size based on your workload
                max_overflow=10,  # Adjust maximum overflow connections
                pool_recycle=3600,  # Periodically recycle connections (optional)
                pool_pre_ping=True,  # Check the connection status before using it
                # poolclass=NullPool,
                # pool_size=self._db_settings.DB_POOL_SIZE,
                # max_overflow=self._db_settings.DB_MAX_OVERFLOW,
                # future=True,
                echo=False,
            )

    def _set_engine_sync(self) -> None:
        self.engine: AsyncEngine = create_async_engine(
            self._dsn,
            pool_pre_ping=True,
            poolclass=NullPool,
            future=True,
            echo=False,
        )

    def _create_session(self) -> None:
        self._async_session_maker = sessionmaker(
            bind=self.engine, expire_on_commit=False, class_=AsyncSession
        )

    def get_sync_session(self):
        return scoped_session(
            sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
            )
        )

    def get_async_session_maker(self) -> sessionmaker:
        return sessionmaker(bind=self.engine, expire_on_commit=False, class_=AsyncSession)

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        self._create_session()
        async with self._async_session_maker() as session:
            yield session

    def sync_run(self) -> None:
        self._set_engine_sync()

    def sync_stop(self) -> None:
        self.engine.dispose()

    def run(self) -> None:
        """
        Create engine
        :return:
        """
        self._set_engine()

    async def stop(self) -> None:
        """
        Destroy engine
        :return:
        """
        await self.engine.dispose()

    async def init_db(self, Base: DeclarativeBase) -> None:
        """use it if u not use alembic"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def delete_db(self, Base: DeclarativeBase) -> None:
        """use it if u not use alembic"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all())

    async def check_connection(self) -> None:
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT version();"))
                version = result.scalar()
                # logger.info(f"PostgreSQL version: {version}")
        except ConnectionRefusedError as e:
            logger.error(f"Failed to connect to the database: {e}")
            raise ConnectionRefusedError(e)

    async def check_alembic_version(self) -> None:
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT version_num FROM alembic_version;"))
                version = result.scalar()
                if version:
                    pass
                    # logger.info(f"Alembic version: {version}")
                else:
                    logger.error("Failed to check Alembic version")

        except ProgrammingError:
            logger.error("No Alembic version found. Migrations may not be applied.")
