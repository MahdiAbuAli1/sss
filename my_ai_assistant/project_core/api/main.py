# project_core/api/main.py

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# --- استيراد الوحدات المخصصة ---
from project_core.core.retrieval import retriever
from project_core.core.config import get_generative_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- إعداد التطبيق والمسجل ---
app = FastAPI(
    title="API المساعد الذكي متعدد الأنظمة",
    description="واجهة برمجة تطبيقات للتحدث مع المساعد الذكي القادر على فهم المستندات العربية.",
    version="2.0.0", # قمنا بترقية الإصدار!
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- نماذج البيانات (Pydantic Models) ---
class QueryRequest(BaseModel):
    question: str
    tenant_id: str

class AnswerResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

# --- تهيئة النموذج اللغوي الكبير ---
try:
    generative_llm = get_generative_llm()
    logger.info("✅ تم تهيئة النموذج اللغوي الكبير (LLM) بنجاح.")
except Exception as e:
    logger.error(f"❌ فشل فادح في تهيئة النموذج اللغوي الكبير: {e}")
    generative_llm = None

# --- **نقطة النهاية الجديدة والمحسّنة مع التوليد** ---
@app.post("/ask", response_model=AnswerResponse, summary="اطرح سؤالاً واحصل على إجابة")
async def ask_assistant(request: QueryRequest):
    if not retriever or not generative_llm:
        raise HTTPException(status_code=500, detail="أحد المكونات الأساسية (Retriever or LLM) غير جاهز.")
    
    logger.info(f"🔍 استلام سؤال جديد للنظام '{request.tenant_id}': '{request.question}'")

    try:
        # --- 1. مرحلة الاسترجاع (Retrieval) ---
        logger.info(f"--- تطبيق مرشح البحث للنظام '{request.tenant_id}' ---")
        session_retriever = retriever.vectorstore.as_retriever(
            search_kwargs={'k': 5, 'filter': {'tenant_id': request.tenant_id}}
        )

        # --- 2. إعداد سلسلة المعالجة (RAG Chain) ---
        template = """
        أنت مساعد ذكي ومحترف. مهمتك هي الإجابة على السؤال التالي بناءً على السياق المقدم فقط.
        إذا كانت المعلومات في السياق غير كافية، قل "المعلومات المتوفرة غير كافية للإجابة على هذا السؤال".
        كن دقيقًا ومختصرًا.

        السياق:
        {context}

        السؤال:
        {question}

        الإجابة المفصلة باللغة العربية:
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            # تنسيق المستندات التي تم العثور عليها في نص واحد
            return "\n\n".join(f"المستند {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

        rag_chain = (
            {"context": session_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | generative_llm
            | StrOutputParser()
        )

        # --- 3. مرحلة التوليد (Generation) ---
        logger.info("...البحث عن المستندات ذات الصلة وتمريرها إلى النموذج اللغوي الكبير لتوليد الإجابة...")
        final_answer = rag_chain.invoke(request.question)
        
        # جلب المستندات المصدر بشكل منفصل لعرضها في الاستجابة
        source_documents = session_retriever.get_relevant_documents(request.question)

        if not source_documents:
             logger.warning(f"⚠️ لم يتم العثور على مستندات مصدر للسؤال.")
             final_answer = "لم أجد معلومات ذات صلة بسؤالك في قاعدة المعرفة المخصصة لهذا النظام."

        logger.info(f"✅ تم توليد الإجابة بنجاح.")

        return AnswerResponse(
            answer=final_answer,
            source_documents=[{"content": doc.page_content, "metadata": doc.metadata} for doc in source_documents]
        )

    except Exception as e:
        logger.error(f"❌ حدث خطأ فادح أثناء معالجة السؤال: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ داخلي في الخادم. التفاصيل: {e}")

@app.get("/", summary="نقطة التحقق من الحالة")
def read_root():
    return {"message": "مرحباً بك في واجهة برمجة تطبيقات المساعد الذكي! الخادم يعمل بشكل صحيح."}
