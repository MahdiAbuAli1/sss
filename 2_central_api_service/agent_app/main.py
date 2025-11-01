# المسار: 2_central_api_service/agent_app/main.py

import os
import logging
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.staticfiles import StaticFiles

# استيراد المنطق الأساسي من ملفنا الآخر
from .core_logic import initialize_agent, get_answer_stream, agent_ready

# --- إعداد التسجيل (Logging) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- الملفات الثابتة ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# --- قراءة إعدادات الأمان من متغيرات البيئة ---
EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY")
if not EXPECTED_API_KEY:
    raise RuntimeError("SUPPORT_SERVICE_API_KEY غير موجود في البيئة. قم بتعيينه قبل التشغيل.")

# --- دليل لحفظ سجلات التفاعلات ---
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
os.makedirs(LOG_DIR, exist_ok=True)

# --- دورة حياة التطبيق (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(" بدء تشغيل خادم الـ API...")
    
    # جدولة التهيئة لتعمل في الخلفية
    loop = asyncio.get_running_loop()
    loop.create_task(initialize_agent())
    logging.info(" تهيئة الوكيل ستعمل في الخلفية...")
    
    yield
    
    logging.info(" إيقاف تشغيل خادم الـ API...")

# --- إنشاء تطبيق FastAPI ---
app = FastAPI(
    title="منصة الدعم الفني المركزي",
    description="واجهة برمجية للوصول إلى وكيل دعم فني متعدد العملاء.",
    version="1.0.0",
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

# Mount static after app creation
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- نماذج البيانات ---
class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    k_results: int = 4
    follow_up_mode: Optional[str] = None  # 'summary' | 'detail'
    follow_up_context: Optional[str] = None

# --- طبقة الأمان ---
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != EXPECTED_API_KEY:
        logging.warning(f"محاولة وصول فاشلة باستخدام مفتاح API غير صحيح: {x_api_key}")
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")
    return x_api_key

# --- نقطة النهاية الرئيسية ---
@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    if not agent_ready():
        raise HTTPException(status_code=503, detail="الخدمة قيد التهيئة حاليًا. يرجى المحاولة بعد لحظات.")
        
    log_file = os.path.join(LOG_DIR, f"{request.tenant_id}_interactions.log")

    async def generator_wrapper() -> AsyncGenerator[str, None]:
        final_answer = ""
        try:
            streamer = get_answer_stream({
                "question": request.question,
                "tenant_id": request.tenant_id,
                "k_results": request.k_results
            })
            async for chunk_data in streamer:
                if chunk_data.get("type") == "error":
                    error_content = json.dumps({"error": chunk_data["content"]}, ensure_ascii=False)
                    yield error_content + "\n"
                    return 
                
                chunk = chunk_data.get("content", "")
                final_answer += chunk
                yield json.dumps({"chunk": chunk}, ensure_ascii=False) + "\n"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- Question ---\n{request.question}\n")
                f.write(f"--- Answer ---\n{final_answer}\n")
                f.write(f"{'='*80}\n\n")

        except Exception as e:
            logging.error(f"خطأ في generator_wrapper: {e}", exc_info=True)
            yield json.dumps({"error": "حدث خطأ داخلي أثناء معالجة طلبك."}, ensure_ascii=False) + "\n"

    return StreamingResponse(generator_wrapper(), media_type="application/x-ndjson")

@app.get("/")
def read_root():
    return {"message": "مرحبًا بك في الواجهة البرمجية لمنصة الدعم الفني المركزي"}

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"status": "ok"}

@app.get("/readyz", response_class=JSONResponse)
def readyz():
    return {"ready": agent_ready()}

@app.get("/tenants", response_class=JSONResponse)
def list_tenants():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../1_knowledge_pipeline/_processing_outputs"))
    tenants = []
    try:
        if os.path.isdir(base_path):
            for name in os.listdir(base_path):
                full = os.path.join(base_path, name)
                if os.path.isdir(full):
                    tenants.append(name)
    except Exception:
        pass
    return {"tenants": tenants}

@app.get("/chat")
def chat_page():
    chat_file = os.path.join(STATIC_DIR, "chat.html")
    if os.path.isfile(chat_file):
        return FileResponse(chat_file)
    return HTMLResponse("تعذر العثور على واجهة المحادثة", status_code=500)

