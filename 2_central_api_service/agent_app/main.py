# # المسار: 2_central_api_service/agent_app/main.py

# import os
# import logging
# from fastapi import FastAPI, Header, HTTPException, Depends
# from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# from typing import AsyncGenerator, Optional
# import asyncio
# from fastapi.middleware.cors import CORSMiddleware
# import json
# from fastapi.staticfiles import StaticFiles

# # استيراد المنطق الأساسي من ملفنا الآخر
# from .core_logic import initialize_agent, get_answer_stream, agent_ready

# # --- إعداد التسجيل (Logging) ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- الملفات الثابتة ---
# STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# # --- قراءة إعدادات الأمان من متغيرات البيئة ---
# EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY")
# if not EXPECTED_API_KEY:
#     raise RuntimeError("SUPPORT_SERVICE_API_KEY غير موجود في البيئة. قم بتعيينه قبل التشغيل.")

# # --- دليل لحفظ سجلات التفاعلات ---
# LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
# os.makedirs(LOG_DIR, exist_ok=True)

# # --- دورة حياة التطبيق (Lifespan) ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logging.info(" بدء تشغيل خادم الـ API...")
    
#     # جدولة التهيئة لتعمل في الخلفية
#     loop = asyncio.get_running_loop()
#     loop.create_task(initialize_agent())
#     logging.info(" تهيئة الوكيل ستعمل في الخلفية...")
    
#     yield
    
#     logging.info(" إيقاف تشغيل خادم الـ API...")

# # --- إنشاء تطبيق FastAPI ---
# app = FastAPI(
#     title="منصة الدعم الفني المركزي",
#     description="واجهة برمجية للوصول إلى وكيل دعم فني متعدد العملاء.",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # --- تمكين CORS ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount static after app creation
# if os.path.isdir(STATIC_DIR):
#     app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # --- نماذج البيانات ---
# class QueryRequest(BaseModel):
#     question: str
#     tenant_id: str
#     k_results: int = 4
#     follow_up_mode: Optional[str] = None  # 'summary' | 'detail'
#     follow_up_context: Optional[str] = None

# # --- طبقة الأمان ---
# async def verify_api_key(x_api_key: str = Header(...)):
#     if x_api_key != EXPECTED_API_KEY:
#         logging.warning(f"محاولة وصول فاشلة باستخدام مفتاح API غير صحيح: {x_api_key}")
#         raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")
#     return x_api_key

# # --- نقطة النهاية الرئيسية ---
# @app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
# async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
#     if not agent_ready():
#         raise HTTPException(status_code=503, detail="الخدمة قيد التهيئة حاليًا. يرجى المحاولة بعد لحظات.")
        
#     log_file = os.path.join(LOG_DIR, f"{request.tenant_id}_interactions.log")

#     async def generator_wrapper() -> AsyncGenerator[str, None]:
#         final_answer = ""
#         try:
#             streamer = get_answer_stream({
#                 "question": request.question,
#                 "tenant_id": request.tenant_id,
#                 "k_results": request.k_results
#             })
#             async for chunk_data in streamer:
#                 if chunk_data.get("type") == "error":
#                     error_content = json.dumps({"error": chunk_data["content"]}, ensure_ascii=False)
#                     yield error_content + "\n"
#                     return 
                
#                 chunk = chunk_data.get("content", "")
#                 final_answer += chunk
#                 yield json.dumps({"chunk": chunk}, ensure_ascii=False) + "\n"

#             with open(log_file, "a", encoding="utf-8") as f:
#                 f.write(f"--- Question ---\n{request.question}\n")
#                 f.write(f"--- Answer ---\n{final_answer}\n")
#                 f.write(f"{'='*80}\n\n")

#         except Exception as e:
#             logging.error(f"خطأ في generator_wrapper: {e}", exc_info=True)
#             yield json.dumps({"error": "حدث خطأ داخلي أثناء معالجة طلبك."}, ensure_ascii=False) + "\n"

#     return StreamingResponse(generator_wrapper(), media_type="application/x-ndjson")

# @app.get("/")
# def read_root():
#     return {"message": "مرحبًا بك في الواجهة البرمجية لمنصة الدعم الفني المركزي"}

# @app.get("/healthz", response_class=JSONResponse)
# def healthz():
#     return {"status": "ok"}

# @app.get("/readyz", response_class=JSONResponse)
# def readyz():
#     return {"ready": agent_ready()}

# @app.get("/tenants", response_class=JSONResponse)
# def list_tenants():
#     base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../1_knowledge_pipeline/_processing_outputs"))
#     tenants = []
#     try:
#         if os.path.isdir(base_path):
#             for name in os.listdir(base_path):
#                 full = os.path.join(base_path, name)
#                 if os.path.isdir(full):
#                     tenants.append(name)
#     except Exception:
#         pass
#     return {"tenants": tenants}

# @app.get("/chat")
# def chat_page():
#     chat_file = os.path.join(STATIC_DIR, "chat.html")
#     if os.path.isfile(chat_file):
#         return FileResponse(chat_file)
#     return HTMLResponse("تعذر العثور على واجهة المحادثة", status_code=500)
# المسار: 2_central_api_service/agent_app/main.py
# --- الإصدار الإنتاجي النهائي (v-Pro-Max-FIXED-v7) ---

# import logging
# import asyncio
# import os
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# import uvicorn

# # --- استيراد العقل الذكي والملفات الشخصية من الحزم الصحيحة ---
# from .core import initialize_agent, get_answer_stream, agent_ready
# from .core.config import SYSTEM_PROFILES

# # --- إعدادات أساسية ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- تهيئة تطبيق FastAPI ---
# app = FastAPI()

# # --- تحديد المسار الصحيح للملفات الثابتة ---
# # هذا يضمن أن الخادم يجد ملف chat.html دائمًا
# static_path = os.path.join(os.path.dirname(__file__), "static")
# app.mount("/static", StaticFiles(directory=static_path), name="static")

# # --- نقطة بداية التطبيق ---
# @app.on_event("startup")
# async def startup_event():
#     """
#     عند بدء تشغيل الخادم، قم بإنشاء مهمة في الخلفية لتهيئة الوكيل.
#     هذا يمنع حجب الخادم ويسمح له بالبدء فورًا.
#     """
#     logging.info("بدء تشغيل خادم الـ API...")
#     asyncio.create_task(initialize_agent())
#     logging.info("تهيئة الوكيل ستعمل في الخلفية...")

# # --- نقطة النهاية الرئيسية لعرض صفحة الدردشة ---
# @app.get("/", response_class=HTMLResponse)
# async def get_chat_page():
#     html_file_path = os.path.join(static_path, "chat.html")
#     try:
#         with open(html_file_path, "r", encoding="utf-8") as f:
#             return HTMLResponse(content=f.read())
#     except FileNotFoundError:
#         return HTMLResponse(content="<h1>Error: chat.html not found</h1>", status_code=404)

# # --- نقطة النهاية لجلب قائمة الأنظمة ---
# @app.get("/tenants", response_class=JSONResponse)
# async def get_tenants():
#     tenants_list = [
#         {"id": key, "name": profile["name"]}
#         for key, profile in SYSTEM_PROFILES.items()
#     ]
#     return {"tenants": tenants_list}

# # --- نقطة النهاية للتحقق من جاهزية الوكيل ---
# @app.get("/health")
# async def health_check():
#     if agent_ready():
#         return {"status": "ready"}
#     return {"status": "initializing"}

# # --- نقطة النهاية الرئيسية للدردشة (WebSocket) ---
# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_json()
#             question = data.get("question")
#             tenant_id = data.get("tenant_id")

#             if not question:
#                 continue

#             if not agent_ready():
#                 await websocket.send_json({"type": "error", "content": "الوكيل لا يزال قيد التهيئة، يرجى الانتظار قليلاً."})
#                 continue

#             # استدعاء العقل الذكي للحصول على الإجابة المتدفقة
#             async for chunk in get_answer_stream({"question": question, "tenant_id": tenant_id, "session_id": session_id}):
#                 await websocket.send_json(chunk)

#     except WebSocketDisconnect:
#         logging.info(f"العميل '{session_id}' قطع الاتصال.")
#     except Exception as e:
#         logging.error(f"حدث خطأ في WebSocket: {e}", exc_info=True)
#         try:
#             await websocket.send_json({"type": "error", "content": "حدث خطأ في الخادم."})
#         except:
#             pass # العميل قد يكون أغلق الاتصال بالفعل
#     finally:
#         if websocket.client_state != 3: # 3 is DISCONNECTED
#              await websocket.close()

# # --- لتشغيل الخادم مباشرة من هذا الملف (اختياري) ---
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# 2_central_api_service/agent_app/main.py (النسخة النهائية v2.0)
# 2_central_api_service/agent_app/main.py (النسخة النهائية v3.0 - متزامنة)

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

# **الإصلاح 1: إزالة استيراد SYSTEM_PROFILES الذي تم حذفه**
from .core_logic import (
    initialize_agent, 
    get_answer_stream, 
    agent_ready,
    # **الإصلاح 2: استيراد القاموس الجديد للردود السريعة بدلاً من ذلك**
    # (ملاحظة: هذا ليس ضروريًا للخادم، لكنه يوضح أننا نستخدم بنية جديدة)
    # سنقوم بجلب البيانات من core_logic ديناميكيًا
)

# --- إعدادات أساسية ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# --- دورة حياة التطبيق (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("بدء تشغيل خادم الـ API...")
    asyncio.create_task(initialize_agent())
    yield
    logging.info("إيقاف تشغيل خادم الـ API...")

# --- إنشاء تطبيق FastAPI ---
app = FastAPI(
    title="منصة الدعم الفني المركزي",
    description="واجهة برمجية وواجهة مستخدم للوصول إلى وكيل دعم فني متعدد العملاء.",
    version="3.0.0",
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
    **الإصلاح 3: تحديث هذه الدالة لتعمل بدون SYSTEM_PROFILES.**
    الآن، تقوم بجلب أسماء العملاء مباشرة من المسترجعات المهيأة.
    """
    from .core_logic import retrievers_cache # استيراد ديناميكي
    
    if not agent_ready():
        logging.warning("الوكيل غير جاهز، سيتم إرجاع قائمة فارغة للعملاء.")
        return []
    
    # بناء أسماء وهمية بسيطة من معرفات العملاء
    profiles = [
        TenantProfile(id=tenant_id, name=f"نظام {tenant_id.replace('_', ' ').title()}")
        for tenant_id in retrievers_cache.keys()
    ]
    return profiles

@app.websocket("/ws/{tenant_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str):
    from .core_logic import retrievers_cache # استيراد ديناميكي
    await websocket.accept()
    
    if tenant_id not in retrievers_cache:
        await websocket.send_json({"type": "error", "content": "النظام المحدد غير صالح."})
        await websocket.close(code=1008)
        return

    logging.info(f"تم إنشاء اتصال WebSocket للعميل: {tenant_id}")

    try:
        while True:
            question = await websocket.receive_text()
            
            if not agent_ready():
                await websocket.send_json({"type": "error", "content": "الوكيل لا يزال قيد التهيئة، يرجى الانتظار."})
                continue

            request_info = {"question": question, "tenant_id": tenant_id}
            
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
