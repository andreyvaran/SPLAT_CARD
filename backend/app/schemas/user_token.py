
from pydantic import UUID4

from app.schemas.base import MyBaseModel

class UserTokenModel(MyBaseModel):
    id: UUID4
    name: str| None = None
    is_active: bool

class UserTokenRequest(MyBaseModel):
    hex_token: str

class CreateOrGetUserTokenRequest(MyBaseModel):
    name: str
    secret: str