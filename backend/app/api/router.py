from fastapi import APIRouter

from config.api import settings
from app.api.v1.research import router as research_router
from app.api.v1.user_token import router as ut_router

router = APIRouter(prefix=settings.APP_PREFIX)


router.include_router(research_router)
router.include_router(ut_router)
