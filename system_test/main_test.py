# system_test/main_test.py

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# --- التعديل الرئيسي: الاستيراد من ملف التحليل الجديد ---
from .core_logic_analyzable import (
    initialize_agent, 
    get_answer_stream, 
    agent_ready,
    get_all_tenants_from_cache # دالة مساعدة جديدة لجلب العملاء
)

# --- إعدادات أساسية ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
# تعديل المسار ليشير إلى المجلد الصحيح داخل بيئة الاختبار
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# --- دورة حياة التطبيق (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("بدء تشغيل خادم الـ API (وضع الاختبار التحليلي)...")
    # تهيئة الوكيل عند بدء التشغيل
    asyncio.create_task(initialize_agent())
    yield
    logging.info("إيقاف تشغيل خادم الـ API (وضع الاختبار التحليلي)...")

# --- إنشاء تطبيق FastAPI ---
app = FastAPI(
    title="منصة الدعم الفني المركزي (وضع الاختبار)",
    description="واجهة برمجية للوصول إلى وكيل الدعم الفني مع تفعيل سجلات التحليل.",
    version="3.0.0-test",
    lifespan=lifespan
)

# --- تمكين CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- تحميل الملفات الثابتة ---
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- نماذج البيانات ---
class TenantProfile(BaseModel):
    id: str
    name: str

# --- نقاط النهاية (API Endpoints) ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_chat_ui():
    chat_file = os.path.join(STATIC_DIR, "chat.html")
    if os.path.isfile(chat_file):
        return FileResponse(chat_file)
    return HTMLResponse("<h1>تعذر العثور على واجهة المحادثة.</h1>", status_code=404)

@app.get("/tenants", response_model=List[TenantProfile])
async def get_tenants():
    """
    يجلب قائمة العملاء المتاحين من ذاكرة التخزين المؤقت للمسترجعات.
    """
    if not agent_ready():
        logging.warning("الوكيل غير جاهز، سيتم إرجاع قائمة فارغة للعملاء.")
        return []
    
    tenant_ids = get_all_tenants_from_cache()
    profiles = [
        TenantProfile(id=tenant_id, name=f"نظام {tenant_id.replace('_', ' ').title()}")
        for tenant_id in tenant_ids
    ]
    return profiles

@app.websocket("/ws/{tenant_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str):
    await websocket.accept()
    
    # التحقق من أن الوكيل جاهز تمامًا قبل قبول أي طلب
    if not agent_ready():
        await websocket.send_json({"type": "error", "content": "الوكيل لا يزال قيد التهيئة، يرجى المحاولة بعد لحظات."})
        await websocket.close(code=1008)
        return

    # التحقق من صلاحية العميل
    if tenant_id not in get_all_tenants_from_cache():
        await websocket.send_json({"type": "error", "content": "النظام المحدد غير صالح."})
        await websocket.close(code=1008)
        return

    logging.info(f"تم إنشاء اتصال WebSocket للعميل: {tenant_id}")

    try:
        while True:
            question = await websocket.receive_text()
            
            request_info = {"question": question, "tenant_id": tenant_id}
            
            # استدعاء الدالة المعدلة التي تحتوي على منطق التحليل
            async for response_chunk in get_answer_stream(request_info):
                await websocket.send_json(response_chunk)
            
            await websocket.send_json({"type": "end_of_stream"})

    except WebSocketDisconnect:
        logging.info(f"تم قطع اتصال WebSocket للعميل: {tenant_id}")
    except Exception as e:
        logging.error(f"حدث خطأ في WebSocket للعميل {tenant_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": "حدث خطأ في الخادم."})
        except RuntimeError:
            pass
