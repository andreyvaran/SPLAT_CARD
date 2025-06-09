import io
import os
import logging
import shutil

import aiofiles
import asyncio
from typing import Any, List
from pydantic import BaseModel
from config.file_storage import FileStorageSettings
from storage.protocol.protocol import Storage, StorageResult


class FileStorageResult(BaseModel):
    data: Any = None
    message: str | None = None



class FileStorage(Storage):
    def __init__(self, settings: FileStorageSettings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.storage_path = settings.root_path

        # Создаем корневую структуру директорий
        os.makedirs(self.storage_path, exist_ok=True)
        self.logger.info(f"FileStorage initialized at {self.storage_path}")

    async def send_to_storage(self, file_contents: bytes | io.BytesIO, file_name: str) -> FileStorageResult:
        try:
            if isinstance(file_contents, io.BytesIO):
                file_contents = file_contents.getvalue()
            clean_file_name = os.path.basename(file_name)
            dest_path = os.path.join(self.storage_path, clean_file_name)

            async with aiofiles.open(dest_path, "wb") as dest_file:
                await dest_file.write(file_contents)

            self.logger.info(f"File saved: {clean_file_name} (size: {len(file_contents)} bytes)")
            return FileStorageResult(
                data=dest_path,
                message="File saved successfully"
            )
        except Exception as e:
            self.logger.error(f"Save error: {str(e)}", exc_info=True)
            return FileStorageResult(
                message=f"Save failed: {str(e)}",
            )


    async def get_from_storage(self, file_name: str) -> FileStorageResult:
        """Возвращает полный путь к файлу в хранилище"""
        file_path = os.path.join(self.storage_path, file_name)

        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_name}")
            return StorageResult(
                message=f"File '{file_name}' not found",
                success=False
            )

        return StorageResult(
            data=file_path,
            message="File found"
        )

    async def delete_by_name(self, file_name: str) -> FileStorageResult:
        file_path = os.path.join(self.storage_path, file_name)
        if not os.path.exists(file_path):
            self.logger.warning(f"Delete failed, file not found: {file_name}")
            return StorageResult(
                message=f"File '{file_name}' not found",
                success=False
            )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, os.remove, file_path)
            self.logger.info(f"File deleted: {file_name}")
            return StorageResult(message=f"File '{file_name}' deleted")
        except Exception as e:
            self.logger.error(f"Delete error: {str(e)}")
            return StorageResult(
                message=f"Delete failed: {str(e)}",
                success=False
            )

    async def list_all(self) -> FileStorageResult:
        try:
            loop = asyncio.get_running_loop()
            files: List[str] = await loop.run_in_executor(
                None,
                lambda: os.listdir(self.storage_path)
            )
            self.logger.info(f"Listed {len(files)} files")
            return StorageResult(data=files, message="Files listed")
        except Exception as e:
            self.logger.error(f"List error: {str(e)}")
            return StorageResult(
                message=f"List failed: {str(e)}",
                success=False
            )

    async def download_all_from_storage(self, target_dir: str) -> FileStorageResult:
        try:
            os.makedirs(target_dir, exist_ok=True)
            loop = asyncio.get_running_loop()

            files: List[str] = await loop.run_in_executor(
                None,
                lambda: os.listdir(self.storage_path)
            )

            for file_name in files:
                src = os.path.join(self.storage_path, file_name)
                dst = os.path.join(target_dir, file_name)
                await loop.run_in_executor(
                    None,
                    lambda s=src, d=dst: shutil.copy2(s, d)
                )

            self.logger.info(f"Downloaded {len(files)} files to {target_dir}")
            return StorageResult(message=f"Files downloaded to {target_dir}")
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return StorageResult(
                message=f"Download failed: {str(e)}",
                success=False
            )

default_file_storage = FileStorage(FileStorageSettings())