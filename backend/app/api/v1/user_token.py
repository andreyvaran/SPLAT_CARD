from fastapi import APIRouter, status, Response

from app.schemas.user_token import UserTokenRequest, CreateOrGetUserTokenRequest
from app.service.user_token import UserTokenService

router = APIRouter(prefix="/user_token", tags=["User Token"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=UserTokenRequest)
async def create(model: CreateOrGetUserTokenRequest, response: Response) -> UserTokenRequest:
    token = await UserTokenService.create_user_token(model)
    response.headers["X-Auth-SPLAT-Token"] = token.hex_token
    return token


@router.get("", status_code=status.HTTP_200_OK, response_model=UserTokenRequest)
async def get_user_token_by_name(
        name: str,
        secret: str,
        response: Response
) -> UserTokenRequest:
    token = await UserTokenService.get_user_token_by_name(name, secret)
    response.headers["X-Auth-SPLAT-Token"] = token.hex_token
    return token
