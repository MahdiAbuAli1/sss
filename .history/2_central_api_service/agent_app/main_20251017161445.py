import os
import logging
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# استيراد المنطق الأساسي من ملفنا الآخر
from .core_logic import initialize_agent, get_answer_stream, format_docs_with_source

# --- إعداد التسجيل (Logging) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- قراءة إعدادات الأمان من متغيرات البيئة ---
EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY", "default_secret_key")

# --- دليل لحفظ سجلات التفاعلات ---
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
os.makedirs(LOG_DIR, exist_ok=True)

# --- دورة حياة التطبيق (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 بدء تشغيل خادم الـ API...")
    initialize_agent()
    yield
    logging.info("🛑 إيقاف تشغيل خادم الـ API...")

# --- إنشاء تطبيق FastAPI ---
app = FastAPI(
    title="منصة الدعم الفني المركزي",
    description="واجهة برمجية للوصول إلى وكيل دعم فني متعدد العملاء.",
    version="1.0.0",
    lifespan=lifespan
)

# --- نماذج البيانات ---
class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    k_results: int = 4

# --- طبقة الأمان: التحقق من API Key ---
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != EXPECTED_API_KEY:
        logging.warning(f"محاولة وصول فاشلة باستخدام مفتاح API غير صحيح: {x_api_key}")
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")
    return x_api_key

# --- نقطة النهاية الرئيسية مع تسجيل التفاعلات ---
@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    try:
        # إنشاء ملف سجل لكل عميل
        log_file = os.path.join(LOG_DIR, f"{request.tenant_id}_interactions.txt")
        
        # متغير لتجميع الإجابة النهائية
        final_answer = ""

        # دالة تغليف البث لتسجيل الإجابة أثناء الإرسال
        async def generator_wrapper() -> AsyncGenerator[str, None]:
            nonlocal final_answer
            async for chunk in get_answer_stream(
                question=request.question,
                tenant_id=request.tenant_id,
                k_results=request.k_results
            ):
                final_answer += chunk
                yield chunk

            # بعد انتهاء البث، حفظ السؤال والإجابة والمصادر في الملف
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- نوع المستخدم (tenant_id) ---\n{request.tenant_id}\n")
                f.write(f"--- السؤال ---\n{request.question}\n")
                f.write(f"--- الإجابة ---\n{final_answer}\n\n")
                f.write(f"{'='*80}\n\n")

        return StreamingResponse(generator_wrapper(), media_type="text/plain")

    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع في نقطة النهاية /ask-stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي في الخادم.")

@app.get("/")
def read_root():
    return {"message": "مرحبًا بك في الواجهة البرمجية لمنصة الدعم الفني المركزي"}
