# backend_project/urls.py

from django.contrib import admin
from django.urls import path, include # أضف include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')), # أي مسار يبدأ بـ /api/ سيتم توجيهه إلى تطبيق api
]
