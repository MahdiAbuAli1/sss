# 2_backend_api/main.py
from fastapi import FastAPI
from typing import List
from .core.logic import get_all_tenants_status
from .models.schemas import TenantStatus
from fastapi.middleware.cors import CORSMiddleware # للسماح للواجهة بالتحدث مع الخادم

app = FastAPI(
    title="Support Service Platform API",
    description="الخادم الخلفي لإدارة خط أنابيب المعرفة.",
    version="1.0.0"
)

# --- إعدادات CORS للسماح بالاتصال من الواجهة الأمامية ---
# (مهم جدًا للبيئة التطويرية)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # في الإنتاج، يجب تحديد دومين الواجهة فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tenants", response_model=List[TenantStatus], tags=["Tenants"])
async def list_tenants():
    """
    جلب قائمة بجميع العملاء وحالتهم الحالية.
    """
    tenants_status = await get_all_tenants_status()
    return tenants_status

# يمكنك إضافة نقطة نهاية بسيطة للترحيب
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Backend API!"}
