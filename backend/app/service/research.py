import io
import os
import uuid
from pathlib import Path

import pillow_heif
from PIL import Image
from alembic.util import status
from fastapi import UploadFile, status, HTTPException
from fastapi.responses import FileResponse

from app.schemas.research import ResearchData, FileStatus
from database.unitofwork import IUnitOfWork, UnitOfWork
from neiron.image_processor import ImageProcessor, get_image_processor
from storage.file_storage.file_storage import default_file_storage
from storage.protocol.protocol import Storage


class ResearchService:

    @staticmethod
    def image_to_bytesio(
            image: Image.Image,
    ) -> io.BytesIO:
        img_bytes = io.BytesIO()

        save_kwargs = {}
        # if format.upper() in ("JPEG", "WEBP"):
        #     save_kwargs["quality"] = quality
        # elif format.upper() == "PNG":
        #     save_kwargs["optimize"] = optimize

        image.save(img_bytes, format="JPEG", **save_kwargs)
        img_bytes.seek(0)

        return img_bytes

    @staticmethod
    async def convert_any_to_jpg(file: UploadFile) -> Image:
        file_data = await file.read()
        file_io = io.BytesIO(file_data)
        if file.content_type in {"application/octet-stream", "image/heif", "image/heif"}:
            heif_file = pillow_heif.read_heif(file_io)
            image = heif_file.to_pillow()
        else:
            image = Image.open(file_io)

        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')

        return image

    @classmethod
    async def send_photo(
            cls,
            user_token: str,
            photo: UploadFile,
            uow: IUnitOfWork = UnitOfWork(),
            storage: Storage = default_file_storage,
            image_processor: ImageProcessor = get_image_processor(),

    ) -> ResearchData:
        if photo.content_type not in [
            "image/jpeg",
            "image/png",
            "image/heif",
            "image/heic",
            "application/octet-stream",
            # "image/gif",
        ]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupportable file type",
            )

        async with uow:
            user_token = await uow.user_token.get_one(hex_token=user_token)
            if not user_token:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
            # process_image
            image = await cls.convert_any_to_jpg(photo)
            research = await uow.research.add_one(data=dict(
                user_token_id=user_token.id,
            )
            )
            f_name = f"row_{research.id}.jpg"
            storage_result = await storage.send_to_storage(file_contents=cls.image_to_bytesio(image), file_name=f_name)

            if storage_result.data:
                await uow.research.edit_one(research.id, data=dict(
                    status=FileStatus.SavedRow,
                    files=[storage_result.data],
                ))

                # process image
                try:
                    if not image_processor.validate(image):
                        await uow.research.edit_one(research.id, data=dict(
                            status=FileStatus.NotValid,
                        ))
                    else:
                        new_image = image_processor.make_results(image=image, research_id=research.id)
                        await uow.research.edit_one(research.id, data=dict(
                            status=FileStatus.Success,
                            result= {"finish": True, "processed_image": str(new_image)},
                            files=[storage_result.data, str(new_image)],
                        ))
                except Exception as e:
                    await uow.research.edit_one(research.id, data=dict(
                        result={"finish": False, "error_message": str(e)},
                        status=FileStatus.Error,
                    ))

            result = await uow.research.get_by_id(research.id)
            return result.get_schema()

    @classmethod
    async def get_all_files_as_responses(cls, research_id: uuid.UUID, uow: IUnitOfWork = UnitOfWork(),
                                         storage: Storage = default_file_storage,
                                         ) -> FileResponse:
        async with uow:
            research = await uow.research.get_by_id(research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        if not research.files:
            raise HTTPException(status_code=404, detail="No files found for this research")

        file_path = research.files[-1]
        return FileResponse(
            file_path,
            filename=os.path.basename(Path(file_path).name),
            media_type="image/jpg"
        )
