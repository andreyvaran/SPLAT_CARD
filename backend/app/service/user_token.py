
from alembic.util import status
from fastapi import status, HTTPException

from app.schemas.user_token import UserTokenRequest, CreateOrGetUserTokenRequest
from database.unitofwork import IUnitOfWork, UnitOfWork


class UserTokenService:



    @classmethod
    async def create_user_token(
            cls,
            model: CreateOrGetUserTokenRequest,
            uow: IUnitOfWork = UnitOfWork(),
    ) -> UserTokenRequest:

        async with uow:
            if model.secret == "Super Secret":
                user = await uow.user_token.get_first_with_filters(name = model.name)
                if user is not None:
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT
                                        , detail="Name already exists")

                user_token = await uow.user_token.add_one(data=dict(name= model.name))
            else:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

            print(user_token)
            print(user_token.get("hex_token"))
            return UserTokenRequest.model_validate(user_token)

    @classmethod
    async def get_user_token_by_name(
            cls,
            name: str,
            secret : str,
            uow: IUnitOfWork = UnitOfWork(),
    ) -> UserTokenRequest:

        async with uow:
            if secret == "Super Secret":
                user = await uow.user_token.get_one(name =name)

                if user is not None:
                    return user.get_token_schema()
                else:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            else:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


