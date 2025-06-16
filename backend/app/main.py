from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from starlette import status

from database.db import database_accessor
from app.api.router import router
from database.admin import create_admin

from config.app import app_config
from redis import asyncio as aioredis


def bind_exceptions(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": str(exc)},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database_accessor.check_connection()
    await database_accessor.check_alembic_version()

    redis = aioredis.from_url(app_config.REDIS_ENDPOINT)
    # FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

    create_admin(app, database_accessor.engine)  # add admin

    yield

    await database_accessor.stop()



def get_app() -> FastAPI:
    app = FastAPI(**app_config.swagger_conf, lifespan=lifespan)
    bind_exceptions(app)
    app.include_router(router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins="*",
        # allow_origins=app_settings.origins,
        allow_credentials=True,
        allow_methods="*",
        allow_headers="*",
    )
    return app


app = get_app()
