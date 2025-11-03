# المسار: 2_central_api_service/agent_app/core/__init__.py
# --- الإصدار v6: الإصلاح الشامل للعقل الهرمي ---

import os
import logging
import asyncio
import json
import random
import re
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flashrank import Ranker, RerankRequest

# --- 1. الإعدادات ---
__all__ = ["initialize_agent", "get_answer_stream", "agent_ready"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
LOGS_DIR = os.path.join(PROJECT_ROOT, "5_analysis_logs")
CANNED_RESPONSES_DIR = os.path.join(PROJECT_ROOT, "2_central_api_service/agent_app/static_responses")

TOP_K = 7
os.makedirs(LOGS_DIR, exist_ok=True)

# --- 2. الصندوق الأسود: مسجل الطلبات (مُحسَّن) ---
class RequestLogger:
    LOG_FILE_PATH = os.path.join(LOGS_DIR, "central_analysis.log")
    _lock = asyncio.Lock()

    def __init__(self, session_id: str):
        self.request_id = str(uuid.uuid4())[:8]
        self.session_id = session_id
        self.log_entries = []
        self.start_time = time.time()

    def log(self, message: str):
        self.log_entries.append(str(message))

    def log_docs(self, docs: List[Document], title: str):
        self.log(f"\n--- {title} (عدد: {len(docs)}) ---")
        if not docs:
            self.log("   -> لا توجد مستندات.")
            return
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'N/A').split(os.sep)[-1]
            content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:90]
            self.log(f"   {i+1}. [المصدر: {source}] -> \"{content_preview}...\"")

    def log_reranked_docs(self, reranked_results, original_docs_map, title: str):
        self.log(f"\n--- {title} (عدد: {len(reranked_results)}) ---")
        if not reranked_results:
            self.log("   -> لا توجد مستندات.")
            return
        for i, res in enumerate(reranked_results):
            doc_id = res.get("id")
            if doc_id is not None and doc_id in original_docs_map:
                doc = original_docs_map[doc_id]
                source = doc.metadata.get('source', 'N/A').split(os.sep)[-1]
                content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:90]
                self.log(f"   {i+1}. [الدرجة: {res['score']:.4f}] [المصدر: {source}] -> \"{content_preview}...\"")

    async def save(self):
        total_time = time.time() - self.start_time
        self.log(f"\n--- الأداء ---")
        self.log(f"⏱️ إجمالي زمن معالجة الطلب: {total_time:.2f} ثانية")
        
        full_report = "\n".join(self.log_entries)
        
        async with self._lock:
            try:
                with open(self.LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write(full_report + "\n\n")
                logging.info(f"✅ تم إضافة سجل الطلب '{self.request_id}' إلى الملف المركزي.")
            except IOError as e:
                logging.error(f"❌ فشل الكتابة إلى ملف السجل المركزي: {e}")

# --- 3. الطبقة 0: الذاكرة الفورية (بمنطق صارم) ---
class InstantMemory:
    def __init__(self):
        self.responses = {}
        self.load_responses()

    def load_responses(self):
        logging.info("🧠 [الطبقة 0] تحميل الردود الفورية (بمنطق صارم)...")
        if not os.path.isdir(CANNED_RESPONSES_DIR):
            logging.warning(f"مجلد الردود الجاهزة '{CANNED_RESPONSES_DIR}' غير موجود.")
            return
        
        count = 0
        for filename in os.listdir(CANNED_RESPONSES_DIR):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(CANNED_RESPONSES_DIR, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            question = item.get("question", "").strip().lower()
                            answers = item.get("answers")
                            if question and answers:
                                self.responses[question] = answers
                                count += 1
                except Exception as e:
                    logging.error(f"فشل تحميل ملف '{filename}': {e}")
        logging.info(f"✅ [الطبقة 0] تم تحميل {count} قاعدة رد فوري.")

    def get_response(self, question: str) -> Optional[str]:
        # تطابق حرفي وصارم 100%
        exact_match = self.responses.get(question.strip().lower())
        if exact_match:
            return random.choice(exact_match)
        return None

# --- 4. الطبقة 1: الحارس السريع (بمنطق مُحسَّن) ---
class FastGatekeeper:
    def is_nonsensical(self, question: str) -> bool:
        q = question.strip()
        # قصير جدًا
        if len(q) < 3: return True
        # حروف مكررة بشكل مبالغ فيه
        if re.search(r'(.)\1{3,}', q): return True
        # لا يحتوي على أي حروف (فقط أرقام ورموز)
        if not re.search(r'[a-zA-Z\u0600-\u06FF]', q): return True
        # نسبة الحروف إلى إجمالي الطول منخفضة جدًا (ضوضاء)
        letters = re.findall(r'[a-zA-Z\u0600-\u06FF]', q)
        if len(letters) / len(q) < 0.5: return True
        return False

# --- 5. قوالب التوجيه (محسّنة) ---
QUESTION_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
"""Your task is to classify the user's question into one of two categories: "specific_query" or "general_chitchat".
- "specific_query": The user is asking a specific question that can likely be answered from a knowledge base (e.g., "how do I reset my password?", "what is max pooling?").
- "general_chitchat": The user is asking a general knowledge question or making a greeting that is not a simple hello/thanks (e.g., "how are you?", "who is the president?", "what is the weather?").

User Question: "{question}"
Category:
"""
)

DYNAMIC_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
"""أنت "مساعد الدعم الذكي". مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصريًا** على "السياق" المقدم لك من قاعدة المعرفة.

**قواعد صارمة:**
1.  **التحية دائمًا:** ابدأ إجابتك بعبارة ترحيبية مناسبة.
2.  **الالتزام المطلق بالسياق:** إذا كانت المعلومات غير موجودة، قل **فقط**: "لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
3.  **التكيف مع مستوى التفصيل المطلوب ({verbosity}):**
    - **"مختصر"**: قدم إجابة موجزة في جملة أو جملتين.
    - **"مفصل"**: قدم إجابة شاملة ومنظمة باستخدام القوائم.
4.  **الاختصار:** لا تذكر أبدًا كلمات مثل "بناءً على السياق" أو "وفقًا للمستندات".
5.  **الخاتمة التفاعلية:** اختتم دائمًا بسؤال تفاعلي، مثل: "هل هناك أي شيء آخر يمكنني مساعدتك به؟".

---
**السياق:**
{context}
---
**سؤال المستخدم:** {question}
---
**مستوى التفصيل المطلوب:** {verbosity}
---
**إجابتك:**
"""
)

# --- 6. المتغيرات العالمية ---
llm_answer: Ollama = None
llm_classifier: Ollama = None
vector_store: FAISS = None
reranker: Ranker = None
all_docs: List[Document] = None
instant_memory: InstantMemory = None
fast_gatekeeper: FastGatekeeper = None
initialization_lock = asyncio.Lock()

# --- 7. دوال التهيئة والتحقق ---
async def initialize_agent():
    global llm_answer, llm_classifier, vector_store, reranker, all_docs, instant_memory, fast_gatekeeper
    async with initialization_lock:
        if agent_ready(): return
        logging.info("--- 🚀 بدء تهيئة العقل الهرمي فائق السرعة (v6) ---")
        try:
            # التأكد من أن المتغيرات البيئية موجودة قبل الاستخدام
            if not all([EMBEDDING_MODEL, CHAT_MODEL, CLASSIFIER_MODEL, OLLAMA_HOST]):
                raise ValueError("أحد متغيرات البيئة المطلوبة للنماذج غير موجود في ملف .env")

            llm_answer = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
            llm_classifier = Ollama(model=CLASSIFIER_MODEL, base_url=OLLAMA_HOST)
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            vector_store = await asyncio.to_thread(FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            reranker = Ranker()
            all_docs = list(vector_store.docstore._dict.values())
            
            instant_memory = InstantMemory()
            fast_gatekeeper = FastGatekeeper()

            logging.info("--- ✅ العقل الذكي جاهز للعمل ---")
        except Exception as e:
            logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
            raise

def agent_ready() -> bool:
    return vector_store is not None and instant_memory is not None

def _get_verbosity(question: str) -> str:
    question_lower = question.lower()
    if any(word in question_lower for word in ["باختصار", "موجز"]):
        return "مختصر"
    return "مفصل"

# --- 8. الدالة الرئيسية لتوليد الإجابة (العقل الهرمي) ---
async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
    question = request_info.get("question", "")
    tenant_id = request_info.get("tenant_id", "default_session")
    session_id = request_info.get("session_id", "default_session")
    logger = RequestLogger(session_id)
    
    logger.log("="*80)
    logger.log(f"طلب جديد | ID: {logger.request_id}")
    logger.log(f"العميل: {tenant_id}")
    logger.log(f"السؤال: {question}")
    logger.log("="*80)

    if not agent_ready():
        yield {"type": "error", "content": "الوكيل غير جاهز."}
        await logger.save()
        return

    try:
        # --- الطبقة 0: الذاكرة الفورية (تطابق حرفي) ---
        logger.log("\n[الطبقة 0: التحقق من الذاكرة الفورية]")
        canned_response = instant_memory.get_response(question)
        if canned_response:
            logger.log(f"-> ✅ تم العثور على رد فوري: '{canned_response}'")
            yield {"type": "full_answer", "content": canned_response}
            return

        # --- الطبقة 1: الحارس السريع (أسئلة تافهة) ---
        logger.log("\n[الطبقة 1: التحقق من حارس البوابة السريع]")
        if fast_gatekeeper.is_nonsensical(question):
            answer = "لم أفهم سؤالك. هل يمكنك إعادة صياغته؟"
            logger.log(f"->  GATEKEEPER: تم تصنيف السؤال على أنه تافه.")
            logger.log(f"\n--- الإجابة النهائية (من حارس البوابة) ---\n{answer}")
            yield {"type": "full_answer", "content": answer}
            return

        # --- الطبقة 2: المصنف الذكي (LLM) ---
        logger.log("\n[الطبقة 2: تصنيف السؤال الذكي]")
        classifier_chain = QUESTION_CLASSIFIER_PROMPT | llm_classifier | StrOutputParser()
        classification_result = await classifier_chain.ainvoke({"question": question})
        classification = re.sub(r'[^a-z_]', '', classification_result.strip().lower())
        logger.log(f"-> التصنيف: {classification}")

        if "general_chitchat" in classification:
            answer = "أنا مساعد متخصص ولا أستطيع الإجابة على أسئلة عامة. هل لديك سؤال حول النظام؟"
            logger.log(f"\n--- الإجابة النهائية (من المصنف) ---\n{answer}")
            yield {"type": "full_answer", "content": answer}
            return

        # --- الطبقة 3: العقل الخارق (RAG) ---
        logger.log("\n[الطبقة 3: بدء عملية الاسترجاع والتوليد (العقل الخارق)]")
        tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
        if not tenant_docs:
            answer = f"لا توجد بيانات للعميل '{tenant_id}'."
            logger.log(f"\n--- خطأ ---\n{answer}")
            yield {"type": "full_answer", "content": answer}
            return

        bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=TOP_K)
        store = InMemoryStore()
        parent_retriever = ParentDocumentRetriever(vectorstore=vector_store, docstore=store, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400))
        parent_retriever.add_documents(tenant_docs, ids=None)
        faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
        hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

        hybrid_docs, parent_docs = await asyncio.gather(
            hybrid_retriever.ainvoke(question),
            asyncio.to_thread(parent_retriever.invoke, question)
        )
        logger.log_docs(hybrid_docs, "المرشحون من البحث الهجين")
        logger.log_docs(parent_docs, "المرشحون من مسترجع المستندات الأصلية")

        combined_docs = hybrid_docs + parent_docs
        unique_docs_map = {doc.page_content: doc for doc in reversed(combined_docs)}
        unique_docs = list(unique_docs_map.values())[::-1]
        logger.log(f"-> إجمالي عدد المرشحين الفريدين: {len(unique_docs)}")

        if not unique_docs:
            answer = "لم يتم العثور على أي معلومات ذات صلة في قاعدة المعرفة."
            yield {"type": "full_answer", "content": answer}
            return

        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(unique_docs)]
        reranked_results = reranker.rerank(RerankRequest(query=question, passages=passages))
        
        top_results = reranked_results[:4]
        original_docs_map = {i: doc for i, doc in enumerate(unique_docs)}
        final_context_docs = [original_docs_map[res["id"]] for res in top_results if res.get("id") in original_docs_map]
        logger.log_reranked_docs(top_results, original_docs_map, "أفضل 4 مستندات بعد إعادة الترتيب")
        
        final_context = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])
        logger.log("\n--- السياق النهائي المرسل للنموذج ---\n" + final_context)

        verbosity = _get_verbosity(question)
        logger.log(f"\n[توليد الإجابة] مستوى التفصيل المطلوب: {verbosity}")
        
        answer_chain = DYNAMIC_PROMPT_TEMPLATE | llm_answer | StrOutputParser()
        
        full_answer = ""
        async for chunk in answer_chain.astream({"context": final_context, "question": question, "verbosity": verbosity}):
            if chunk:
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}
        
        logger.log("\n--- الإجابة النهائية ---\n" + full_answer)

    except Exception as e:
        error_message = "عذراً، حدث خطأ فادح أثناء معالجة طلبك."
        logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        logger.log(f"\n--- خطأ فادح ---\n{e}")
        yield {"type": "error", "content": error_message}
    finally:
        await logger.save()
