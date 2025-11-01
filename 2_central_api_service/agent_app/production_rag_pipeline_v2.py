# المسار: 2_central_api_service/agent_app/production_rag_pipeline_v2.py
# --- خط الأنابيب الإنتاجي الكامل مع "العقل المرن" ---
#لحكم النهائي على الكود v2
#الكود v2 ينتج إجابات عالية الجودة ولكنه غير صالح للاستخدام الإنتاجي على الإطلاق بسبب بطئه الشديد. لقد نجحنا في بناء "عقل ذكي" ولكنه "عقل بطيء".
#
import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv

# --- (جميع استيرادات المكونات تبقى كما هي) ---
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

# --- (جميع الإعدادات الأساسية تبقى كما هي) ---
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
TOP_K = 7

# --- 3. قالب التوجيه الديناميكي (v2) ---
DYNAMIC_PROMPT_TEMPLATE = """
أنت "مساعد الدعم الذكي". مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصريًا** على "السياق" المقدم لك من قاعدة المعرفة.

**شخصيتك:**
- **خبير وموثوق:** واثق من معلوماتك ودقيق. لا تخترع أي إجابات.
- **مساعد ومرن:** هدفك هو مساعدة المستخدم بالطريقة التي يفضلها.

**قواعد صارمة لا يمكن كسرها:**
1.  **التحية دائمًا:** ابدأ إجابتك دائمًا بعبارة ترحيبية مناسبة (مثال: "أهلاً بك!"، "بالتأكيد، بخصوص سؤالك...").
2.  **الالتزام المطلق بالسياق:**
    - إذا كانت المعلومات موجودة في السياق، أجب عليها.
    - إذا كانت المعلومات غير موجودة **تمامًا** في السياق، يجب أن تقول **فقط**: "لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
3.  **التكيف مع مستوى التفصيل المطلوب ({verbosity}):**
    - إذا كان مستوى التفصيل المطلوب هو **"مختصر"**: قدم إجابة مباشرة وموجزة في جملة أو جملتين.
    - إذا كان مستوى التفصيل المطلوب هو **"مفصل"**: قدم إجابة شاملة ومنظمة. استخدم القوائم النقطية أو الرقمية لتوضيح الخطوات أو النقاط المتعددة.
4.  **الاختصار:** لا تذكر أبدًا كلمات مثل "بناءً على السياق" أو "وفقًا للمستندات".
5.  **الخاتمة التفاعلية:** اختتم دائمًا بسؤال تفاعلي، مثل: "هل هناك أي شيء آخر يمكنني مساعدتك به؟".

---
**السياق من قاعدة المعرفة (مصدر الحقيقة الوحيد):**
{context}
---
**سؤال المستخدم:**
{question}
---
**مستوى التفصيل المطلوب:**
{verbosity}
---
**إجابتك (مع الالتزام بالتحية، السياق، مستوى التفصيل، والخاتمة):**
"""

# --- 4. الفئة الإنتاجية (مع منطق تحديد التفصيل) ---
class RAGPipeline:
    def __init__(self):
        # ... (التهيئة تبقى كما هي) ...
        print("--- 🚀 تهيئة خط أنابيب RAG الإنتاجي (v2) ---")
        self.llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        self.vector_store = FAISS.load_local(UNIFIED_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        self.reranker = Ranker()
        
        all_docs = list(self.vector_store.docstore._dict.values())
        self.all_tenant_docs = {}
        for doc in all_docs:
            tenant_id = doc.metadata.get("tenant_id")
            if tenant_id:
                if tenant_id not in self.all_tenant_docs:
                    self.all_tenant_docs[tenant_id] = []
                self.all_tenant_docs[tenant_id].append(doc)
        
        self.final_prompt = ChatPromptTemplate.from_template(DYNAMIC_PROMPT_TEMPLATE)
        self.answer_chain = self.final_prompt | self.llm | StrOutputParser()
        print("--- ✅ خط الأنابيب جاهز ---")

    def _get_verbosity(self, question: str) -> str:
        """يحدد مستوى التفصيل المطلوب بناءً على كلمات في السؤال."""
        question_lower = question.lower()
        if any(word in question_lower for word in ["باختصار", "موجز", "هل يمكن"]):
            return "مختصر"
        # "اشرح"، "بالتفصيل"، "ما هو" ستؤدي إلى الخيار الافتراضي
        return "مفصل"

    async def get_answer(self, question: str, tenant_id: str) -> str:
        print(f"\n[>>] تم استلام سؤال جديد للعميل '{tenant_id}': '{question}'")

        # --- المرحلة 1: الاسترجاع باستخدام "المسترجع الشامل" ---
        # ... (كود الاسترجاع الشامل يبقى كما هو دون تغيير) ...
        print("[1/2] 🔍 تنفيذ الاسترجاع الشامل...")
        tenant_docs = self.all_tenant_docs.get(tenant_id)
        if not tenant_docs: return "خطأ: لا توجد بيانات لهذا العميل."
        faiss_retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
        bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=TOP_K)
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        store = InMemoryStore()
        parent_document_retriever = ParentDocumentRetriever(vectorstore=self.vector_store, docstore=store, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400))
        parent_document_retriever.add_documents(tenant_docs, ids=None)
        hybrid_docs = await ensemble_retriever.ainvoke(question)
        parent_docs = await asyncio.to_thread(parent_document_retriever.invoke, question)
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
        
        # --- المرحلة 2: تحديد مستوى التفصيل وتوليد الإجابة ---
        verbosity = self._get_verbosity(question)
        print(f"[2/2] 🧠 توليد الإجابة (مستوى التفصيل: {verbosity})...")
        
        final_answer = await self.answer_chain.ainvoke({
            "context": final_context,
            "question": question,
            "verbosity": verbosity  # تمرير مستوى التفصيل إلى القالب
        })
        
        return final_answer

# --- 5. نقطة الدخول الرئيسية للتجربة (مع أسئلة جديدة لاختبار المرونة) ---
async def main():
    pipeline = RAGPipeline()

    # --- أسئلة جديدة لاختبار الذكاء الحواري ---
    test_cases = [
        # اختبار الشرح المفصل (الافتراضي)
        {"tenant_id": "school_beta", "question": "اشرح لي ما هي طبقة الـ pooling في الشبكات العصبية؟"},
        # اختبار الإجابة المختصرة
        {"tenant_id": "school_beta", "question": "باختصار، ما هو الغرض من طبقة الـ pooling؟"},
        # اختبار عدم وجود معلومات (الهلوسة)
        {"tenant_id": "un", "question": "ما هو سعر سهم شركة أبل اليوم؟"},
        # اختبار سؤال مركب (يتطلب تفصيل)
        {"tenant_id": "university_alpha", "question": "ما هي أهداف مشروع Plant Care وكيف يختلف عن تطبيق Plantix؟"}
    ]

    for case in test_cases:
        answer = await pipeline.get_answer(question=case["question"], tenant_id=case["tenant_id"])
        print("\n" + "="*30 + " 💬 الإجابة النهائية 💬 " + "="*30)
        print(f"السؤال: {case['question']}")
        print(f"الإجابة:\n{answer}")
        print("="*86)

if __name__ == "__main__":
    asyncio.run(main())
