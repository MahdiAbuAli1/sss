# system_test/core_logic_analyzable.py (النسخة النهائية مع جدار العزل)

import os
import logging
import asyncio
import json
import random
import time
import uuid
from typing import AsyncGenerator, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# --- التعديل 1: استيراد CrossEncoder ---
from sentence_transformers.cross_encoder import CrossEncoder

# --- 1. الإعدادات ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

# --- إعداد Logger التحليل ---
analysis_logger = logging.getLogger('AnalysisLogger')
analysis_logger.setLevel(logging.INFO)
analysis_logger.propagate = False
if not analysis_logger.handlers:
    log_directory = os.path.dirname(__file__)
    log_file_path = os.path.join(log_directory, "analysis_log.jsonl")
    print(f"--- DIAGNOSTIC: Log file will be created at: {log_file_path} ---")
    handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    analysis_logger.addHandler(handler)

# --- إعدادات النماذج والمسارات ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# --- التعديل 2: إضافة نموذج CrossEncoder ---
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
HIERARCHICAL_DB_PATH = os.path.join(PROJECT_ROOT, "2_central_api_service", "agent_app", "hierarchical_db.json")

TOP_K = 7
MIN_QUESTION_LENGTH = 3
# --- التعديل 3: إضافة حد أدنى لدرجة الصلة ---
RELEVANCE_THRESHOLD = 0.3

# --- 2. القالب النهائي ---
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """<|system|>
أنت مساعد دعم فني متخصص. مهمتك هي الإجابة على أسئلة المستخدم حول النظام بالاعتماد **حصريًا** على "السياق" المقدم.
- إذا كان السياق يحتوي على إجابة، قدمها مباشرة.
- إذا كان السياق فارغًا أو غير مرتبط بالسؤال، أو إذا كان السؤال عامًا ولا يتعلق بالنظام (مثل "من هو ميسي؟" أو "ما هي عاصمة فرنسا؟")، فيجب أن تكون إجابتك **فقط إحدى هاتين الجملتين**:
  1. إذا كان السؤال لا يتعلق بالنظام: "أنا مساعد دعم فني متخصص، ولا يمكنني الإجابة على أسئلة عامة."
  2. إذا كان السؤال يتعلق بالنظام ولكن لا توجد معلومات: "بخصوص سؤالك '{input}'، لا توجد لدي معلومات كافية في قاعدة المعرفة حاليًا."
- لا تخترع أي إجابات أبدًا.

<|user|>
السياق:
{context}

السؤال: {input}

<|assistant|>
الإجابة:"""
)

# --- 3. المتغيرات العالمية (Cache) ---
llm: Ollama = None
cross_encoder: CrossEncoder = None # إضافة CrossEncoder إلى الذاكرة المؤقتة
vector_store: FAISS = None
retrievers_cache: Dict[str, EnsembleRetriever] = {}
input_map: Dict[str, str] = {}
response_map: Dict[str, List[str]] = {}
concept_to_inputs_map: Dict[str, List[str]] = {}
initialization_lock = asyncio.Lock()

# --- 4. دوال التهيئة والمساعدة ---
async def initialize_agent():
    global llm, cross_encoder, vector_store, retrievers_cache, input_map, response_map, concept_to_inputs_map
    async with initialization_lock:
        if llm is not None: return
        logging.info("🚀 بدء التهيئة الشاملة للوكيل (v-final)...")
        try:
            # تهيئة النماذج بالتوازي
            llm_task = asyncio.to_thread(Ollama, model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
            cross_encoder_task = asyncio.to_thread(CrossEncoder, CROSS_ENCODER_MODEL)
            embeddings_task = asyncio.to_thread(HuggingFaceEmbeddings, model_name=EMBEDDING_MODEL_NAME)
            
            llm, cross_encoder, embeddings = await asyncio.gather(llm_task, cross_encoder_task, embeddings_task)
            logging.info("✅ تم تهيئة نماذج LLM, CrossEncoder, و Embeddings بنجاح.")

            vector_store = await asyncio.to_thread(
                FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            logging.info("✅ تم تحميل قاعدة البيانات المتجهة بنجاح.")

            # ... (بقية كود التهيئة يبقى كما هو)
            all_docs = list(vector_store.docstore._dict.values())
            tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
            for tenant_id in tenants:
                tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
                bm25_retriever = BM25Retriever.from_documents(tenant_docs)
                faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
                retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
            logging.info("✅ تم بناء المسترجعات الهجينة.")

            if os.path.exists(HIERARCHICAL_DB_PATH):
                with open(HIERARCHICAL_DB_PATH, 'r', encoding='utf-8') as f:
                    db_data = json.load(f)
                    input_map = db_data.get("input_map", {})
                    response_map = db_data.get("response_map", {})
                for inp, concept in input_map.items():
                    if concept not in concept_to_inputs_map: concept_to_inputs_map[concept] = []
                    concept_to_inputs_map[concept].append(inp)
                logging.info("⚡ تم تحميل قاعدة البيانات الهرمية بنجاح.")
            else:
                logging.warning(f"⚠️ تحذير: ملف قاعدة البيانات الهرمية غير موجود.")

            logging.info("✅ الوكيل جاهز للعمل بكامل طاقته (مع جدار العزل).")
        except Exception as e:
            logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
            raise

# ... (بقية الدوال المساعدة تبقى كما هي)
def agent_ready(): return llm is not None
def get_all_tenants_from_cache(): return list(retrievers_cache.keys())
def smart_match(q):
    nq = q.lower().strip()
    if nq in input_map: return input_map[nq]
    for cid, inps in concept_to_inputs_map.items():
        for kw in inps:
            if len(kw) >= 3 and kw in nq: return cid
    return None

# --- 5. الدالة الرئيسية لتوليد الإجابة (النسخة النهائية المصححة) ---
async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
    start_time = time.time()
    request_id = str(uuid.uuid4())
    question = request_info.get("question", "").strip()
    tenant_id = request_info.get("tenant_id", "unknown_session")

    analysis_data = { "request_id": request_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tenant_id": tenant_id, "question": question, "processing_path": "N/A", "total_duration_ms": 0, "steps": {}, "final_answer": "", "error": None }

    def finalize_analysis(data):
        end_time = time.time()
        data["total_duration_ms"] = round((end_time - start_time) * 1000)
        log_entry = json.dumps(data, ensure_ascii=False, indent=2)
        analysis_logger.info(log_entry)

    try:
        # --- البوابة 1: جدار الصدّ الذكي ---
        if len(question) < MIN_QUESTION_LENGTH:
            analysis_data["processing_path"] = "rejected_short"
            response = "عذرًا، لم أفهم سؤالك. هل يمكنك توضيحه أكثر؟"
            analysis_data["final_answer"] = response
            yield {"type": "chunk", "content": response}
            return

        # --- البوابة 2: محرك الحوارات الهرمي ---
        concept_id = smart_match(question)
        if concept_id and concept_id in response_map:
            analysis_data["processing_path"] = "fast_path"
            response = random.choice(response_map[concept_id])
            analysis_data["final_answer"] = response
            yield {"type": "chunk", "content": response}
            return

        # --- المسار الافتراضي: محرك RAG المعرفي ---
        analysis_data["processing_path"] = "rag_path"
        retriever = retrievers_cache.get(tenant_id)
        if not retriever: raise ValueError(f"لا يوجد مسترجع للعميل '{tenant_id}'.")

        # --- خطوة الاسترجاع ---
        docs = await retriever.ainvoke(question)
        analysis_data["steps"]["2_retrieval"] = { "retrieved_count_initial": len(docs) }

        # --- التعديل 4: بوابة التحقق من الصلة (جدار العزل) ---
        if not docs:
            analysis_data["processing_path"] = "rag_path_no_docs"
            # لا توجد مستندات، دع النموذج يرد بالرسالة الافتراضية
            final_docs = []
        else:
            # إنشاء أزواج [سؤال, محتوى مستند] للتحقق
            pairs = [[question, doc.page_content] for doc in docs]
            scores = await asyncio.to_thread(cross_encoder.predict, pairs)
            
            # إضافة الدرجات إلى المستندات وتصفيتها
            relevant_docs = []
            for i, doc in enumerate(docs):
                doc.metadata['relevance_score'] = float(scores[i])
                if scores[i] >= RELEVANCE_THRESHOLD:
                    relevant_docs.append(doc)
            
            # فرز المستندات ذات الصلة حسب الدرجة
            relevant_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
            final_docs = relevant_docs

            analysis_data["steps"]["3_relevance_check"] = {
                "scores": [float(s) for s in scores],
                "threshold": RELEVANCE_THRESHOLD,
                "relevant_count": len(final_docs)
            }

        # --- خطوة التوليد ---
        # إذا لم تمر أي مستندات من بوابة التحقق، سيكون final_docs فارغًا
        # وسيعتمد النموذج على تعليمات البرومبت للرد بشكل صحيح
        answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        full_answer = ""
        async for chunk in answer_chain.astream({"input": question, "context": final_docs}):
            if chunk:
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}
        
        analysis_data["final_answer"] = full_answer.strip()

    except Exception as e:
        error_msg = f"فشل في سلسلة المعالجة: {str(e)}"
        logging.error(f"[{tenant_id}][{request_id}] {error_msg}", exc_info=True)
        analysis_data["error"] = error_msg
        try: yield {"type": "error", "content": "عذرًا، حدث خطأ فادح."}
        except Exception: pass
    finally:
        finalize_analysis(analysis_data)
