# المسار: 2_central_api_service/agent_app/production_rag_pipeline_v3.py
# --- خط الأنابيب الإنتاجي فائق السرعة مع التصنيف المسبق والتخزين المؤقت ---

import asyncio
import os
import time
import re

from typing import Dict, List
from dotenv import load_dotenv

# --- 1. استيراد المكونات ---
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flashrank import Ranker, RerankRequest

# --- 2. الإعدادات الأساسية ---
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL_NAME", "qwen2:1.5b-instruct-q4_K_M") # نموذج صغير وسريع للتصنيف
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
TOP_K = 7

# --- 3. قوالب التوجيه (Prompts) ---

# قالب "حارس البوابة" لتصنيف الأسئلة
QUESTION_CLASSIFIER_PROMPT = """
Your task is to classify the user's question into one of three categories: "specific_query", "general_chitchat", or "nonsensical".
- "specific_query": The user is asking a specific question that can likely be answered from a knowledge base (e.g., "how do I reset my password?", "what is max pooling?").
- "general_chitchat": The user is asking a general knowledge question or making a greeting (e.g., "hello", "who is the president?", "what is the weather?").
- "nonsensical": The user's input is random characters, gibberish, or makes no sense (e.g., "asdfgh", "blablabla", "qwertyy").

User Question: "{question}"
Category:
"""

# قالب الإجابة الديناميكي (لا تغيير)
DYNAMIC_PROMPT_TEMPLATE = """
أنت "مساعد الدعم الذكي"... 
(نفس القالب من v2)
"""

# --- 4. فئة إدارة المسترجعات (مع التخزين المؤقت) ---
class RetrieverManager:
    """
    فئة مسؤولة عن إنشاء وإدارة وتخزين المسترجعات بشكل مؤقت لتجنب إعادة البناء المكلفة.
    """
    def __init__(self, vector_store: FAISS, all_tenant_docs: Dict[str, List[Document]]):
        self._vector_store = vector_store
        self._all_tenant_docs = all_tenant_docs
        self._retriever_cache: Dict[str, Dict[str, any]] = {}
        print("🧠 مدير المسترجعات: بدء بناء وتخزين المسترجعات الأولية...")
        self._build_cache()

    def _build_cache(self):
        """يقوم ببناء وتخزين كائنات BM25 و ParentDocument لكل عميل عند بدء التشغيل."""
        for tenant_id, docs in self._all_tenant_docs.items():
            if tenant_id not in self._retriever_cache:
                self._retriever_cache[tenant_id] = {}
            
            # تخزين BM25Retriever
            self._retriever_cache[tenant_id]['bm25'] = BM25Retriever.from_documents(docs, k=TOP_K)
            
            # تخزين ParentDocumentRetriever
            store = InMemoryStore()
            parent_retriever = ParentDocumentRetriever(
                vectorstore=self._vector_store, 
                docstore=store, 
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=400)
            )
            parent_retriever.add_documents(docs, ids=None)
            self._retriever_cache[tenant_id]['parent'] = parent_retriever
        print(f"✅ مدير المسترجعات: تم تخزين المسترجعات لـ {len(self._retriever_cache)} عميل.")

    def get_retrievers(self, tenant_id: str) -> Dict[str, any]:
        """
        يُرجع المسترجعات المخزنة مؤقتًا للعميل المحدد.
        """
        if tenant_id not in self._retriever_cache:
            raise ValueError(f"لا توجد مسترجعات مخزنة للعميل: {tenant_id}")
            
        # المسترجعات التي تعتمد على الفلترة (سريعة ولا تحتاج لتخزين)
        faiss_retriever = self._vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}}
        )
        
        # استدعاء المسترجعات من الذاكرة المؤقتة
        bm25_retriever = self._retriever_cache[tenant_id]['bm25']
        parent_retriever = self._retriever_cache[tenant_id]['parent']
        
        # بناء المسترجع الهجين عند الطلب (سريع)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], 
            weights=[0.5, 0.5]
        )
        
        return {
            "hybrid": ensemble_retriever,
            "parent": parent_retriever
        }

# --- 5. الفئة الإنتاجية النهائية (v3) - سريعة ومنظمة ---
class RAGPipeline:
    def __init__(self):
        print("--- 🚀 تهيئة خط أنابيب RAG فائق السرعة (v3) ---")
        # نماذج اللغة
        self.answer_llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
        self.classifier_llm = Ollama(model=CLASSIFIER_MODEL, base_url=OLLAMA_HOST)
        
        # أدوات الاسترجاع
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        self.vector_store = FAISS.load_local(UNIFIED_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        self.reranker = Ranker()
        
        # تحميل البيانات وإعداد مدير المسترجعات (مع التخزين المؤقت)
        all_docs = list(self.vector_store.docstore._dict.values())
        all_tenant_docs = self._group_docs_by_tenant(all_docs)
        self.retriever_manager = RetrieverManager(self.vector_store, all_tenant_docs)
        
        # سلاسل المعالجة
        self.classifier_prompt = ChatPromptTemplate.from_template(QUESTION_CLASSIFIER_PROMPT)
        self.classifier_chain = self.classifier_prompt | self.classifier_llm | StrOutputParser()
        
        self.final_prompt = ChatPromptTemplate.from_template(DYNAMIC_PROMPT_TEMPLATE)
        self.answer_chain = self.final_prompt | self.answer_llm | StrOutputParser()
        print("--- ✅ خط الأنابيب جاهز للعمل ---")

    def _group_docs_by_tenant(self, all_docs: List[Document]) -> Dict[str, List[Document]]:
        """يجمع المستندات حسب هوية العميل."""
        grouped = {}
        for doc in all_docs:
            tenant_id = doc.metadata.get("tenant_id")
            if tenant_id:
                if tenant_id not in grouped:
                    grouped[tenant_id] = []
                grouped[tenant_id].append(doc)
        return grouped

    def _get_verbosity(self, question: str) -> str:
        """يحدد مستوى التفصيل."""
        # ... (نفس الدالة من v2) ...
        question_lower = question.lower()
        if any(word in question_lower for word in ["باختصار", "موجز", "هل يمكن"]):
            return "مختصر"
        return "مفصل"

    async def get_answer(self, question: str, tenant_id: str) -> str:
        print(f"\n[>>] تم استلام سؤال جديد للعميل '{tenant_id}': '{question}'")

        # --- المرحلة 1: التصنيف المسبق (حارس البوابة) ---
        print("[1/3] 🛡️ تصنيف نية المستخدم...")
        classification_result = await self.classifier_chain.ainvoke({"question": question})
        classification = classification_result.strip().lower()
        print(f"   -> التصنيف: {classification}")

        if classification == "general_chitchat":
            return "أنا مساعد متخصص في أنظمة محددة ولا أستطيع الإجابة على أسئلة عامة. هل لديك سؤال حول النظام؟"
        if classification == "nonsensical":
            return "لم أفهم سؤالك. هل يمكنك إعادة صياغته؟"

        # --- المرحلة 2: الاسترجاع الشامل (فائق السرعة) ---
        print("[2/3] 🔍 تنفيذ الاسترجاع الشامل (باستخدام الذاكرة المؤقتة)...")
        try:
            retrievers = self.retriever_manager.get_retrievers(tenant_id)
            hybrid_retriever = retrievers['hybrid']
            parent_retriever = retrievers['parent']
            
            # تشغيل المسترجعات بالتوازي
            hybrid_docs, parent_docs = await asyncio.gather(
                hybrid_retriever.ainvoke(question),
                asyncio.to_thread(parent_retriever.invoke, question)
            )
        except Exception as e:
            return f"حدث خطأ أثناء استرجاع البيانات: {e}"

        # ... (بقية منطق الدمج وإعادة الترتيب يبقى كما هو) ...
        combined_initial_docs = hybrid_docs + parent_docs
        unique_docs_map = {doc.page_content: doc for doc in reversed(combined_initial_docs)}
        unique_docs = list(unique_docs_map.values())[::-1]
        if not unique_docs: return "لم يتم العثور على أي معلومات ذات صلة."
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(unique_docs)]
        reranked_results = self.reranker.rerank(RerankRequest(query=question, passages=passages))
        top_results = reranked_results[:4]
        original_docs_map = {i: doc for i, doc in enumerate(unique_docs)}
        final_context_docs = [original_docs_map[res["id"]] for res in top_results]
        final_context = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])
        
        # --- المرحلة 3: توليد الإجابة ---
        verbosity = self._get_verbosity(question)
        print(f"[3/3] 🧠 توليد الإجابة (مستوى التفصيل: {verbosity})...")
        
        final_answer = await self.answer_chain.ainvoke({
            "context": final_context,
            "question": question,
            "verbosity": verbosity
        })
        
        return final_answer

# --- 6. نقطة الدخول الرئيسية للتجربة ---
async def main():
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"فشل في تهيئة خط الأنابيب: {e}")
        return

    # --- أسئلة لاختبار السرعة والذكاء ---
    test_cases = [
        # اختبار السرعة لسؤال عادي
        {"tenant_id": "school_beta", "question": "اشرح لي ما هي طبقة الـ pooling في الشبكات العصبية؟"},
        # اختبار حارس البوابة (سؤال عام)
        {"tenant_id": "un", "question": "مرحباً، كيف حالك اليوم؟"},
        # اختبار حارس البوابة (سؤال غير منطقي)
        {"tenant_id": "sys", "question": "بلبلبلبلبب"},
        # اختبار سؤال مركب (للتأكد من أن الجودة لم تتأثر)
        {"tenant_id": "university_alpha", "question": "باختصار، كيف يساهم التطبيق في تحقيق عائد مالي للمزارعين؟"}
    ]

    for case in test_cases:
        start_time = time.time()
        answer = await pipeline.get_answer(question=case["question"], tenant_id=case["tenant_id"])
        duration = time.time() - start_time
        
        print("\n" + "="*30 + " 💬 الإجابة النهائية 💬 " + "="*30)
        print(f"السؤال: {case['question']}")
        print(f"الإجابة:\n{answer}")
        print(f"⏱️ إجمالي زمن الاستجابة: {duration:.2f} ثانية")
        print("="*86)

if __name__ == "__main__":
    asyncio.run(main())
