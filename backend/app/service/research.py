import io
import os
import uuid
from pathlib import Path

import pillow_heif
from alembic.util import status
from fastapi import UploadFile, status, HTTPException
from fastapi.responses import FileResponse

from app.schemas.research import ResearchData, FileStatus
from database.unitofwork import IUnitOfWork, UnitOfWork
from storage.file_storage.file_storage import default_file_storage
from storage.protocol.protocol import Storage


class ResearchService:

    @staticmethod
    async def convert_any_to_jpg(file: UploadFile) -> io.BytesIO:
        if file.content_type != "application/octet-stream":
            file_data = await file.read()
            return io.BytesIO(
                file_data
            )  # Возвращаем оригинальные данные в формате BytesIO

        file_data = await file.read()

        heif_file = pillow_heif.read_heif(io.BytesIO(file_data))
        image = heif_file.to_pillow()
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        return img_bytes

    @classmethod
    async def send_photo(
            cls,
            user_token: str,
            photo: UploadFile,
            uow: IUnitOfWork = UnitOfWork(),
            storage : Storage = default_file_storage,
    ) -> ResearchData:
        if photo.content_type not in [
            "image/jpeg",
            "image/png",
            "image/heif",
            "image/heic",
            "application/octet-stream",
            "image/gif",
        ]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupportable file type",
            )

        async with uow:
            user_token = await uow.user_token.get_one(hex_token=user_token)
            print(user_token)
            if not user_token:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
            # process_image
            file_data = await cls.convert_any_to_jpg(photo)
            research = await uow.research.add_one(data=dict(
                user_token_id=user_token.id,
            )
            )
            f_name = f"row_{research.id}.jpg"
            storage_result = await storage.send_to_storage(file_contents=file_data, file_name=f_name)
            print(storage_result)
            if storage_result.data:
                await uow.research.edit_one(research.id, data = dict(
                    status = FileStatus.SavedRow,
                    files = [storage_result.data],
                ))
            else:
                await uow.research.edit_one(research.id, data=dict(
                    status=FileStatus.Error,
                ))
            result = await uow.research.get_by_id(research.id)
            return result.get_schema()

    @classmethod
    async def get_all_files_as_responses(cls, research_id: uuid.UUID,uow: IUnitOfWork = UnitOfWork(),
                                         storage: Storage = default_file_storage,
                                         ) -> FileResponse:
        async with uow:
            research = await uow.research.get_by_id(research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        if not research.files:
            raise HTTPException(status_code=404, detail="No files found for this research")
        # file_names = [f"raw_{research.id}.jpg"]
        # responses = []
        # for filename in file_names:
        #     storage_data = await storage.get_from_storage(filename)
        #     if storage_data.data:
        #
        #         response = FileResponse(
        #             storage_data.data,
        #             filename=os.path.basename(filename),
        #             media_type="image/jpg"
        #         )
        #         responses.append(response)
        for file_path in research.files:
            return  FileResponse(
                        file_path,
                        filename=os.path.basename(Path(file_path).name),
                        media_type="image/jpg"
                    )

