from database.models import UserTokenORM
from database.repository.repository import SQLAlchemyRepository


class UserTokenRepository(SQLAlchemyRepository):
    model = UserTokenORM

