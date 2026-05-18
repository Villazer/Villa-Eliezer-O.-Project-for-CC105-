from django.urls import path
from . import views
urlpatterns = [
    path('',                      views.index,        name='index'),
    path('classify/',             views.classify,     name='classify'),
    path('logs/',                 views.logs,         name='logs'),
    path('logs/<int:pk>/',        views.log_detail,   name='log_detail'),
    path('logs/<int:pk>/delete/', views.log_delete,   name='log_delete'),
    path('manage-model/',         views.manage_model, name='manage_model'),
]
