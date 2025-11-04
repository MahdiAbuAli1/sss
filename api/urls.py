# api/urls.py (ملف جديد)

from django.urls import path
from .views import TenantStatusList

urlpatterns = [
    path('tenants/', TenantStatusList.as_view(), name='tenant-status-list'),
]
