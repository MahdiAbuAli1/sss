# 2_central_api_service/agent_app/main.py
# -----------------------------------------------------------------------------
# هذا هو خادم الـ API المركزي باستخدام FastAPI.
# يوفر نقطة نهاية آمنة وتفاعلية (streaming) للوصول إلى وكيل الدعم الفني.
# -----------------------------------------------------------------------------

import os
import logging
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# استيراد المنطق الأساسي من ملفنا الآخر
from .core_logic import initialize_agent, get_answer_stream

# --- إعداد التسجيل (Logging) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- قراءة إعدادات الأمان من متغيرات البيئة ---
# هذا هو المفتاح السري الذي يجب على العملاء إرساله للوصول إلى الخدمة
EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY", "default_secret_key")

# --- دورة حياة التطبيق (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # هذا الكود يتم تنفيذه مرة واحدة فقط عند بدء تشغيل الخادم
    logging.info("🚀 بدء تشغيل خادم الـ API...")
    initialize_agent()
    yield
    # هذا الكود يتم تنفيذه عند إيقاف الخادم (غير مستخدم حاليًا)
    logging.info("🛑 إيقاف تشغيل خادم الـ API...")

# --- إنشاء تطبيق FastAPI مع دورة الحياة ---
app = FastAPI(
    title="منصة الدعم الفني المركزي",
    description="واجهة برمجية للوصول إلى وكيل دعم فني متعدد العملاء.",
    version="1.0.0",
    lifespan=lifespan
)

# --- نماذج البيانات (للتدقيق والتحقق من صحة الطلبات) ---
class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    k_results: int = 4 # قيمة افتراضية يمكن تغييرها في الطلب

# --- طبقة الأمان: دالة للتحقق من مفتاح الـ API ---
async def verify_api_key(x_api_key: str = Header(...)):
    """
    يتحقق من أن مفتاح الـ API المرسل في الـ Header صحيح.
    إذا لم يكن صحيحًا، يثير خطأ HTTP 401.
    """
    if x_api_key != EXPECTED_API_KEY:
        logging.warning(f"محاولة وصول فاشلة باستخدام مفتاح API غير صحيح: {x_api_key}")
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")
    return x_api_key

# --- نقطة النهاية الرئيسية (Endpoint) ---
@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    """
    نقطة نهاية تفاعلية (streaming) للإجابة على أسئلة العملاء.
    1. تتحقق من صحة مفتاح الـ API.
    2. تستقبل السؤال وهوية العميل.
    3. تبث الإجابة كلمة بكلمة.
    """
    try:
        # استدعاء دالة البث من core_logic
        answer_generator = get_answer_stream(
            question=request.question,
            tenant_id=request.tenant_id,
            k_results=request.k_results
        )
        # إرجاع استجابة تفاعلية
        return StreamingResponse(answer_generator, media_type="text/plain")
    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع في نقطة النهاية /ask-stream: {e}", exc_info=True)
        # إرجاع خطأ 500 إذا حدث أي شيء خاطئ
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي في الخادم.")

@app.get("/")
def read_root():
    return {"message": "مرحبًا بك في الواجهة البرمجية لمنصة الدعم الفني المركزي"}

