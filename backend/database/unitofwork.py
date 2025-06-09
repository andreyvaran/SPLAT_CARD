"""Module implementing Unit of Work pattern for managing repositories."""

from abc import ABC, abstractmethod
from typing import Type

from database.db import database_accessor

from database.repository import ResearchRepository, UserTokenRepository


class IUnitOfWork(ABC):
    """Interface for Unit of Work pattern."""

    user_token: Type[UserTokenRepository]
    research: Type[ResearchRepository]

    @abstractmethod
    def __init__(self):
        """Initialize the Unit of Work instance."""

    @abstractmethod
    async def __aenter__(self):
        """Enter the context manager."""

    @abstractmethod
    async def __aexit__(self, *args):
        """Exit the context manager."""

    @abstractmethod
    async def commit(self):
        """Commit changes."""

    @abstractmethod
    async def rollback(self):
        """Rollback changes."""


class UnitOfWork:
    def __init__(self):
        self.session_fabric = database_accessor.get_async_session_maker()

    async def __aenter__(self):
        """Enter the context manager."""
        self.session = self.session_fabric()

        self.user_token = UserTokenRepository(self.session)
        self.research = ResearchRepository(self.session)

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
        await self.session.close()

    async def commit(self):
        await self.session.commit()

    async def rollback(self):
        await self.session.rollback()
