import uuid

from fastapi import APIRouter, status, UploadFile, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from app.schemas.research import ResearchData
from app.service.research import ResearchService

router = APIRouter(prefix="/research", tags=["Crud to themes"])


async def get_token(x_auth_token: str = Header(..., alias="X-Auth-SPLAT-Token")) -> str:
    if not x_auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Auth-Token header"
        )
    return x_auth_token


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ResearchData)
async def create(photo: UploadFile, token: str = Depends(get_token)) -> ResearchData:
    return await ResearchService.send_photo(token, photo)


@router.get("", status_code=status.HTTP_201_CREATED)
async def get_all_research_files(research: uuid.UUID, token: str = Depends(get_token)) -> FileResponse:
    return await ResearchService.get_all_files_as_responses(research)
