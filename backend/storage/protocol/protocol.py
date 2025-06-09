from typing import Protocol, Any

from pydantic import BaseModel


class StorageResult(BaseModel):
    data: Any = None
    message: str | None = None

class Storage(Protocol):
    async def send_to_storage(self, file_contents: bytes, file_name: str) -> StorageResult:
        raise NotImplementedError("Subclasses must implement send_to_storage method.")

    async def get_from_storage(self, file: str) -> StorageResult:
        raise NotImplementedError("Subclasses must implement get_from_storage method.")

    async def get_url(self, file: str) -> str:
        raise NotImplementedError("Subclasses must implement get_url method.")

    async def get_storage_list(self) -> list:
        raise NotImplementedError("Subclasses must implement get_storage_list method.")

    async def delete_from_storage(self, file: str) -> str:
        raise NotImplementedError("Subclasses must implement delete_from_storage method.")
