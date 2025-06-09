from fastapi import APIRouter, status

from app.schemas.user_token import UserTokenRequest, CreateOrGetUserTokenRequest
from app.service.user_token import UserTokenService

router = APIRouter(prefix="/user_token", tags=["User Token"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=UserTokenRequest)
async def create(model: CreateOrGetUserTokenRequest) -> UserTokenRequest:
    return await UserTokenService.create_user_token(model)

@router.get("", status_code=status.HTTP_200_OK, response_model=UserTokenRequest)
async def get_user_token_by_name(name: str,
    secret: str) -> UserTokenRequest:
    return await UserTokenService.get_user_token_by_name(name, secret)

