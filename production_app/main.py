# production_app/main.py

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# لا تقم بتغيير هذه الاستيرادات
from .core.agent import agent_instance
from .core.models import TenantProfile
# لا نعتمد على config.STATIC_DIR هنا بعد الآن

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 بدء تشغيل خادم الإنتاج...")
    asyncio.create_task(agent_instance.initialize())
    yield
    logging.info("⛔ إيقاف تشغيل خادم الإنتاج.")

app = FastAPI(title="منصة الدعم الفني (إنتاج)", version="4.0.0", lifespan=lifespan)

# --- Middlewares ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- التعديل الجوهري هنا ---
# هذا هو الكود الصحيح لتحميل الملفات الساكنة بشكل مقاوم للأخطاء
# 1. احصل على المسار المطلق للمجلد الذي يحتوي على هذا الملف (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. ابنِ المسار إلى مجلد "static" الذي يقع بجواره
static_path = os.path.join(current_dir, "static")
# 3. قم بتوصيل المسار "/static" بالمجلد الفعلي على القرص الصلب
app.mount("/static", StaticFiles(directory=static_path), name="static")
# ----------------------------------------------------

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_chat_ui():
    # استخدم المسار الذي قمنا ببنائه للوصول إلى chat.html
    chat_file_path = os.path.join(static_path, "chat.html")
    return FileResponse(chat_file_path)

@app.get("/tenants")
async def get_tenants():
    if not agent_instance.is_ready():
        return []
    profiles = [
        TenantProfile(id=tenant_id, name=f"نظام {tenant_id.replace('_', ' ').title()}")
        for tenant_id in agent_instance.get_tenants()
    ]
    return profiles

@app.websocket("/ws/{tenant_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, session_id: str):
    await websocket.accept()
    if not agent_instance.is_ready():
        await websocket.send_json({"type": "error", "content": "الوكيل لا يزال قيد التهيئة، يرجى الانتظار."})
        await websocket.close()
        return
    
    if tenant_id not in agent_instance.get_tenants():
        await websocket.send_json({"type": "error", "content": "النظام المحدد غير صالح."})
        await websocket.close()
        return

    logging.info(f"تم إنشاء اتصال WebSocket للجلسة: {session_id}")
    try:
        while True:
            question = await websocket.receive_text()
            request_data = {
                "question": question,
                "tenant_id": tenant_id,
                "session_id": session_id,
            }
            async for chunk in agent_instance.get_answer_stream(request_data):
                await websocket.send_json(chunk)
            await websocket.send_json({"type": "end_of_stream"})
    except WebSocketDisconnect:
        logging.info(f"تم قطع الاتصال للجلسة: {session_id}")
    except Exception as e:
        logging.error(f"خطأ في WebSocket للجلسة {session_id}: {e}", exc_info=True)

