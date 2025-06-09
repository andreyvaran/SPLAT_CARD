import uuid

from fastapi import APIRouter, status, UploadFile
from fastapi.responses import FileResponse
from app.schemas.research import ResearchData
from app.service.research import ResearchService

router = APIRouter(prefix="/research", tags=["Crud to themes"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ResearchData)
async def create(user_token: str, photo: UploadFile) -> ResearchData:
    return await ResearchService.send_photo(user_token, photo)

@router.get("", status_code=status.HTTP_201_CREATED)
async def get_all_research_files(research: uuid.UUID)-> FileResponse:
    return await ResearchService.get_all_files_as_responses(research)

