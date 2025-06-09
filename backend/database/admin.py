from sqladmin import Admin
from sqladmin import ModelView
from .models import (
    UserTokenORM,
    ResearchORM,
)


class BaseModelView(ModelView):
    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True
    can_export = True
    page_size = 100


class UserTokenAdmin(BaseModelView, model=UserTokenORM):
    name = "Токен пользователя"
    name_plural = "Токены пользователей"
    column_list = [
        UserTokenORM.id,
        UserTokenORM.name,
        UserTokenORM.hex_token,
        UserTokenORM.is_active,
    ]
    column_searchable_list = [UserTokenORM.name]
    column_sortable_list = [UserTokenORM.name, UserTokenORM.is_active]
    column_details_exclude_list = [UserTokenORM.researches]
    form_excluded_columns = [UserTokenORM.researches]


class ResearchAdmin(BaseModelView, model=ResearchORM):
    name = "Исследование"
    name_plural = "Исследования"
    column_list = [
        ResearchORM.id,
        ResearchORM.files,
        ResearchORM.create_date,
        ResearchORM.user_token,
    ]
    column_sortable_list = [ResearchORM.create_date, ResearchORM.files]
    column_details_list = [
        ResearchORM.id,
        ResearchORM.files,
        ResearchORM.create_date,
        ResearchORM.result,
        ResearchORM.user_token,
    ]
    form_ajax_refs = {
        'user_token': {
            'fields': (UserTokenORM.name, UserTokenORM.hex_token),
            'order_by': UserTokenORM.name,
        }
    }


def create_admin(app, engine):
    admin = Admin(app, engine, title="Исследования")
    admin.add_view(UserTokenAdmin)
    admin.add_view(ResearchAdmin)
    return admin