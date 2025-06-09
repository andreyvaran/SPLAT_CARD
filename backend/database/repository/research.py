

from database.models import ResearchORM
from database.repository.repository import SQLAlchemyRepository


class ResearchRepository(SQLAlchemyRepository):
    model = ResearchORM


    # async def get_newest_articles(self, limit: int) -> list[ArticleModel]:
    #     query = select(ArticleORM).order_by(ArticleORM.release_date.desc()).limit(limit)
    #     result = await self.session.execute(query)
    #     articles = result.scalars().all()
    #     return [ArticleModel(
    #         id=article.id,
    #         header=article.header,
    #         img_s3_preview=article.img_s3_preview,
    #         release_date=article.release_date,
    #         is_active=article.is_active,
    #         themes=[ThemeModel(
    #             id=theme.id,
    #             name=theme.name,
    #             description=theme.description,
    #             is_active=theme.is_active
    #         ) for theme in article.admin_themes]
    #     ) for article in articles]

