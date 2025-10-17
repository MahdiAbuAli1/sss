# 2_central_api_service/agent_app/core_logic.py (النسخة الاحترافية النهائية)

import os
import logging
from typing import List, Dict, Any, AsyncGenerator
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import langchain
from langchain.cache import InMemoryCache

# --- تفعيل الذاكرة المؤقتة (Cache) ---
logging.info("🚀 تفعيل الذاكرة المؤقتة (InMemoryCache) لـ LangChain...")
langchain.llm_cache = InMemoryCache()

# --- الإعدادات الأولية ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))

# --- قراءة الإعدادات من متغيرات البيئة ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))

# --- قالب الأسئلة المحسن ---
RAG_PROMPT_TEMPLATE = """
**مهمتك:** أنت مساعد دعم فني خبير ومختص. استخدم المعلومات المتوفرة في "السياق" التالي للإجابة على "سؤال المستخدم" بدقة واحترافية.
- السياق المقدم عبارة عن مجموعة من المستندات ذات الصلة.
- إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال." ولا تحاول اختلاق إجابة.
- أجب دائمًا باللغة العربية.

**السياق:**
{context}

**سؤال المستخدم:**
{question}

**الإجابة:**
"""

# --- متغيرات عالمية ---
vector_store = None
llm = None
prompt = None

def initialize_agent():
    """ تقوم بتحميل قاعدة المعرفة والنماذج. تُستدعى مرة واحدة عند بدء تشغيل الـ API. """
    global vector_store, llm, prompt
    
    if vector_store:
        logging.info("الوكيل مُهيأ بالفعل.")
        return

    try:
        logging.info("="*50)
        logging.info("🚀 بدء تهيئة وكيل الدعم الفني...")
        
        # 1. تحميل نموذج التضمين
        logging.info(f"تحميل نموذج التضمين: {EMBEDDING_MODEL_NAME}...")
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

        # 2. تحميل قاعدة بيانات المتجهات FAISS
        logging.info(f"تحميل قاعدة المعرفة من: {VECTOR_DB_PATH}...")
        if not os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
            raise FileNotFoundError(f"قاعدة المعرفة (index.faiss) غير موجودة في المسار: {VECTOR_DB_PATH}. يرجى تشغيل خط أنابيب البيانات أولاً.")
        
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        logging.info("✅ تم تحميل قاعدة المعرفة بنجاح.")

        # 3. تحميل النموذج اللغوي الكبير للمحادثة مع إعدادات إضافية
        logging.info(f"تحميل نموذج المحادثة: {CHAT_MODEL_NAME}...")
        llm = Ollama(
            model=CHAT_MODEL_NAME,
            temperature=0.1,  # تقليل العشوائية لجعل الإجابات أكثر اتساقًا
            # يمكنك إضافة المزيد من الإعدادات هنا مثل top_p, top_k
        )

        # 4. إعداد قالب الأسئلة
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        logging.info("✅ اكتملت تهيئة وكيل الدعم الفني بنجاح!")
        logging.info("="*50)
    except FileNotFoundError as e:
        logging.critical(f"❌ فشل التهيئة: ملف قاعدة المعرفة غير موجود. {e}", exc_info=True)
        raise
    except Exception as e:
        logging.critical(f"❌ فشل فادح وغير متوقع أثناء تهيئة الوكيل: {e}", exc_info=True)
        raise

def format_docs_with_source(docs: List[Dict[str, Any]]) -> str:
    """ دالة مساعدة محسنة: تنسق المستندات مع ذكر مصدرها. """
    if not docs:
        return "لا يوجد سياق متوفر."
    
    sources = {doc.metadata.get('source', 'مصدر غير معروف') for doc in docs}
    formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
    """
    تستقبل سؤالاً وهوية العميل، وتستخدم سلسلة RAG لبث الإجابة بشكل تفاعلي.
    """
    if not vector_store or not llm or not prompt:
        raise RuntimeError("الوكيل غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")
    
    logging.info(f"استقبال طلب بث للعميل '{tenant_id}' (k={k_results}): '{question}'")
    
    try:
        # --- إعداد المسترد (Retriever) مع فلترة ديناميكية ---
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k_results, 'filter': {'tenant_id': tenant_id}}
        )
        
        # --- بناء سلسلة RAG ---
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.get_relevant_documents(x["question"])
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs_with_source(x["context"])
            )
            | prompt
            | llm
        )

        logging.info(f"جارٍ البحث عن إجابة ضمن نطاق العميل '{tenant_id}'...")
        
        # --- البث التفاعلي (Streaming) ---
        async for chunk in rag_chain.astream({"question": question}):
            yield chunk
            
    except Exception as e:
        logging.error(f"حدث خطأ أثناء بث الإجابة للعميل '{tenant_id}': {e}", exc_info=True)
        yield "عذرًا، حدث خطأ داخلي أثناء محاولة الإجابة على سؤالك."

