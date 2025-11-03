# # #C:\Users\mahdi\support_service_platform\1_knowledge_pipeline\comprehensive_lab.py
# # # --- مختبر المواجهة النهائية: 7 طرق استرجاع في اختبار شامل ---

# # import asyncio
# # import os
# # import time
# # from typing import List, Dict, Set
# # from dotenv import load_dotenv

# # # --- 1. استيراد المكونات ---
# # from langchain_core.documents import Document
# # from langchain_community.embeddings import OllamaEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain.retrievers import BM25Retriever, EnsembleRetriever
# # from langchain.storage import InMemoryStore
# # from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from flashrank import Ranker, RerankRequest

# # # --- 2. الإعدادات الأساسية ---
# # load_dotenv()
# # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# # OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
# # TOP_K = 7

# # # --- 3. دالة مساعدة لعرض النتائج (لا تغيير) ---
# # def print_results(docs: List[Document], title: str, duration: float, scores: List[float] = None):
# #     print("\n" + "="*80)
# #     print(f"🔬 نتائج طريقة: {title}")
# #     print(f"⏱️ زمن الاسترجاع: {duration:.4f} ثانية")
# #     print(f"📄 عدد النتائج: {len(docs)}")
# #     print("="*80)
# #     if not docs:
# #         print("   -> لم يتم العثور على نتائج.")
# #         return
# #     for i, doc in enumerate(docs):
# #         source = doc.metadata.get('source', 'غير معروف').split('\\')[-1]
# #         content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:110]
# #         score_info = f"[الدرجة: {scores[i]:.4f}]" if scores and i < len(scores) else ""
# #         print(f"   {i+1}. {score_info} [المصدر: {source}] -> \"{content_preview}...\"")
# #     print("-" * 80)

# # # --- 4. المختبر الرئيسي ---
# # async def run_final_showdown_lab(question: str, tenant_id: str, embeddings: OllamaEmbeddings, vector_store: FAISS, reranker: Ranker, all_tenant_docs: Dict[str, List[Document]]):
# #     print("\n" + "#"*30 + f" بدء الاختبار للعميل: '{tenant_id}' | السؤال: '{question}' " + "#"*30)
    
# #     tenant_docs = all_tenant_docs.get(tenant_id)
# #     if not tenant_docs:
# #         print(f"❌ خطأ: لا توجد مستندات للعميل '{tenant_id}'.")
# #         return

# #     # --- إعداد المسترجعات ---
# #     faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
# #     bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=TOP_K)
# #     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
# #     store = InMemoryStore()
# #     parent_document_retriever = ParentDocumentRetriever(vectorstore=vector_store, docstore=store, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400))
# #     parent_document_retriever.add_documents(tenant_docs, ids=None)

# #     # --- تنفيذ الاختبارات والمقارنة ---
    
# #     # 1. BM25 (Keywords)
# #     start_time = time.time(); bm25_docs = await bm25_retriever.ainvoke(question); duration = time.time() - start_time
# #     print_results(bm25_docs, "1. البحث بالكلمات المفتاحية (BM25)", duration)

# #     # 2. Vector Search
# #     start_time = time.time(); vector_docs = await faiss_retriever.ainvoke(question); duration = time.time() - start_time
# #     print_results(vector_docs, "2. البحث بالمعنى (Vector Search)", duration)

# #     # 3. Hybrid
# #     start_time = time.time(); hybrid_docs = await ensemble_retriever.ainvoke(question); duration = time.time() - start_time
# #     print_results(hybrid_docs, "3. البحث الهجين (Hybrid)", duration)

# #     # 4. Hybrid + Reranker
# #     if hybrid_docs:
# #         start_time = time.time()
# #         passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(hybrid_docs)]
# #         reranked_results = reranker.rerank(RerankRequest(query=question, passages=passages))
# #         duration = time.time() - start_time
# #         original_docs_map = {i: doc for i, doc in enumerate(hybrid_docs)}
# #         final_docs = [original_docs_map[res["id"]] for res in reranked_results]
# #         final_scores = [res["score"] for res in reranked_results]
# #         print_results(final_docs, "4. البحث الهجين + إعادة الترتيب (Hybrid + Reranker)", duration, scores=final_scores)

# #     # 5. Parent Document
# #     start_time = time.time(); parent_docs = await asyncio.to_thread(parent_document_retriever.invoke, question); duration = time.time() - start_time
# #     print_results(parent_docs, "5. مسترجع المستندات الأصلية (Parent Document)", duration)

# #     # 6. Parent + Reranker
# #     if parent_docs:
# #         start_time = time.time()
# #         passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(parent_docs)]
# #         reranked_results = reranker.rerank(RerankRequest(query=question, passages=passages))
# #         duration = time.time() - start_time
# #         original_docs_map = {i: doc for i, doc in enumerate(parent_docs)}
# #         super_hybrid_docs = [original_docs_map[res["id"]] for res in reranked_results]
# #         super_hybrid_scores = [res["score"] for res in reranked_results]
# #         print_results(super_hybrid_docs, "6. المسترجع الفائق (Parent + Reranker)", duration, scores=super_hybrid_scores)

# #     # 7. المسترجع الشامل (Hybrid + Parent + Reranker)
# #     # دمج نتائج البحث الهجين والمستندات الأصلية
# #     combined_initial_docs = hybrid_docs + parent_docs
# #     # إزالة التكرار مع الحفاظ على الترتيب
# #     unique_docs_map = {doc.page_content: doc for doc in reversed(combined_initial_docs)}
# #     unique_docs = list(unique_docs_map.values())[::-1]
    
# #     if unique_docs:
# #         start_time = time.time()
# #         passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(unique_docs)]
# #         reranked_results = reranker.rerank(RerankRequest(query=question, passages=passages))
# #         duration = time.time() - start_time
# #         original_docs_map = {i: doc for i, doc in enumerate(unique_docs)}
# #         ultimate_docs = [original_docs_map[res["id"]] for res in reranked_results]
# #         ultimate_scores = [res["score"] for res in reranked_results]
# #         print_results(ultimate_docs, "7. المسترجع الشامل (Hybrid + Parent + Reranker)", duration, scores=ultimate_scores)


# # # --- 5. الدالة الرئيسية للتنفيذ ---
# # async def main():
# #     print("--- 🚀 بدء تهيئة مختبر المواجهة النهائية 🚀 ---")
# #     try:
# #         embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
# #         vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
# #         reranker = Ranker()
        
# #         all_docs = list(vector_store.docstore._dict.values())
# #         all_tenant_docs = {}
# #         for doc in all_docs:
# #             tenant_id = doc.metadata.get("tenant_id")
# #             if tenant_id:
# #                 if tenant_id not in all_tenant_docs:
# #                     all_tenant_docs[tenant_id] = []
# #                 all_tenant_docs[tenant_id].append(doc)
# #         print("--- ✅ البيئة جاهزة. ---")
# #     except Exception as e:
# #         print(f"❌ فشل فادح في تهيئة البيئة: {e}")
# #         return

# #     # --- تعريف حالات الاختبار (نفس الأسئلة العميقة) ---
# #     test_cases = [
# #         {
# #             "tenant_id": "sys",
# #             "question": "ما هي الإجراءات التصحيحية المطلوبة بعد تقرير الزيارة الميدانية؟"
# #         },
# #         {
# #             "tenant_id": "un",
# #             "question": "ماذا يحدث بعد تقديم العطاء وقبل إرساء العقد؟"
# #         },
# #         {
# #             "tenant_id": "school_beta",
# #             "question": "قارن بين طبقة التجميع الأقصى (Max Pooling) والتجميع المتوسط (Average Pooling)."
# #         },
# #         {
# #             "tenant_id": "university_alpha",
# #             "question": "كيف يساهم التطبيق في تحقيق عائد مالي للمزارعين وما هي حدوده؟"
# #         }
# #     ]

# #     # --- تشغيل جميع حالات الاختبار ---
# #     for case in test_cases:
# #         await run_final_showdown_lab(
# #             question=case["question"],
# #             tenant_id=case["tenant_id"],
# #             embeddings=embeddings,
# #             vector_store=vector_store,
# #             reranker=reranker,
# #             all_tenant_docs=all_tenant_docs
# #         )

# # if __name__ == "__main__":
# #     asyncio.run(main())
# # --- مختبر المواجهة النهائية المحسّن: الإصدار 2.0 ---

# import asyncio
# import os
# import time
# from typing import List, Dict
# from dotenv import load_dotenv

# # --- 1. استيراد المكونات ---
# from langchain_core.documents import Document
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
# from langchain.storage import InMemoryStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # **التحسين: استخدام CrossEncoder من sentence-transformers لإعادة الترتيب**
# from sentence_transformers.cross_encoder import CrossEncoder

# # --- 2. الإعدادات الأساسية ---
# load_dotenv()
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
# TOP_K = 7
# # **التحسين: تحديد نموذج Reranker قوي**
# RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# # --- 3. دالة مساعدة لعرض النتائج (مُحسّنة لعرض الدرجات بشكل أفضل) ---
# def print_results(docs: List[Document], title: str, duration: float, scores: List[float] = None):
#     print("\n" + "="*80)
#     print(f"🔬 نتائج طريقة: {title}")
#     print(f"⏱️ زمن الاسترجاع + إعادة الترتيب: {duration:.4f} ثانية")
#     print(f"📄 عدد النتائج: {len(docs)}")
#     print("="*80)
#     if not docs:
#         print("   -> لم يتم العثور على نتائج.")
#         return
#     for i, doc in enumerate(docs):
#         source = doc.metadata.get('source', 'غير معروف').split('\\')[-1]
#         content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:110]
#         score_info = f"[الدرجة: {scores[i]:.4f}]" if scores and i < len(scores) else ""
#         print(f"   {i+1}. {score_info} [المصدر: {source}] -> \"{content_preview}...\"")
#     print("-" * 80)

# # --- 4. المختبر الرئيسي المحسّن ---
# async def run_final_showdown_lab(
#     question: str,
#     tenant_id: str,
#     vector_store: FAISS,
#     reranker: CrossEncoder,
#     retrievers_cache: Dict[str, Dict]
# ):
#     print("\n" + "#"*30 + f" بدء الاختبار للعميل: '{tenant_id}' | السؤال: '{question}' " + "#"*30)

#     # --- استرجاع المسترجعات المهيأة مسبقًا من الذاكرة المؤقتة ---
#     tenant_retrievers = retrievers_cache.get(tenant_id)
#     if not tenant_retrievers:
#         print(f"❌ خطأ: لا توجد مسترجعات مهيأة للعميل '{tenant_id}'.")
#         return

#     faiss_retriever = tenant_retrievers['faiss']
#     bm25_retriever = tenant_retrievers['bm25']
#     ensemble_retriever = tenant_retrievers['ensemble']
#     parent_document_retriever = tenant_retrievers['parent']

#     # --- تنفيذ الاختبارات والمقارنة ---

#     # 1. BM25 (Keywords)
#     start_time = time.time(); bm25_docs = await bm25_retriever.ainvoke(question); duration = time.time() - start_time
#     print_results(bm25_docs, "1. البحث بالكلمات المفتاحية (BM25)", duration)

#     # 2. Vector Search
#     start_time = time.time(); vector_docs = await faiss_retriever.ainvoke(question); duration = time.time() - start_time
#     print_results(vector_docs, "2. البحث بالمعنى (Vector Search)", duration)

#     # 3. Hybrid (70% Vector, 30% BM25)
#     start_time = time.time(); hybrid_docs = await ensemble_retriever.ainvoke(question); duration = time.time() - start_time
#     print_results(hybrid_docs, "3. البحث الهجين (Hybrid - 70/30)", duration)

#     # 4. Hybrid + Reranker (محسّن)
#     if hybrid_docs:
#         start_time = time.time()
#         # إنشاء أزواج من [السؤال, المحتوى] لـ CrossEncoder
#         passages_for_reranking = [[question, doc.page_content] for doc in hybrid_docs]
#         # حساب الدرجات
#         reranked_scores = reranker.predict(passages_for_reranking)
#         duration = time.time() - start_time
#         # دمج المستندات مع درجاتها الجديدة وترتيبها
#         reranked_hybrid_docs = sorted(zip(hybrid_docs, reranked_scores), key=lambda x: x[1], reverse=True)
#         final_docs = [doc for doc, score in reranked_hybrid_docs]
#         final_scores = [score for doc, score in reranked_hybrid_docs]
#         print_results(final_docs, "4. البحث الهجين + إعادة الترتيب (Hybrid + Reranker)", duration, scores=final_scores)

#     # 5. Parent Document
#     start_time = time.time(); parent_docs = await asyncio.to_thread(parent_document_retriever.invoke, question); duration = time.time() - start_time
#     print_results(parent_docs, "5. مسترجع المستندات الأصلية (Parent Document)", duration)

#     # 6. Parent + Reranker (محسّن)
#     if parent_docs:
#         start_time = time.time()
#         passages_for_reranking = [[question, doc.page_content] for doc in parent_docs]
#         reranked_scores = reranker.predict(passages_for_reranking)
#         duration = time.time() - start_time
#         reranked_parent_docs = sorted(zip(parent_docs, reranked_scores), key=lambda x: x[1], reverse=True)
#         final_docs = [doc for doc, score in reranked_parent_docs]
#         final_scores = [score for doc, score in reranked_parent_docs]
#         print_results(final_docs, "6. المسترجع الفائق (Parent + Reranker)", duration, scores=final_scores)

#     # 7. المسترجع الشامل (Hybrid + Parent + Reranker) (محسّن)
#     combined_initial_docs = hybrid_docs + parent_docs
#     unique_docs_map = {doc.page_content: doc for doc in reversed(combined_initial_docs)}
#     unique_docs = list(unique_docs_map.values())[::-1]

#     if unique_docs:
#         start_time = time.time()
#         passages_for_reranking = [[question, doc.page_content] for doc in unique_docs]
#         reranked_scores = reranker.predict(passages_for_reranking)
#         duration = time.time() - start_time
#         reranked_ultimate_docs = sorted(zip(unique_docs, reranked_scores), key=lambda x: x[1], reverse=True)
#         ultimate_docs = [doc for doc, score in reranked_ultimate_docs]
#         ultimate_scores = [score for doc, score in reranked_ultimate_docs]
#         print_results(ultimate_docs, "7. المسترجع الشامل (Hybrid + Parent + Reranker)", duration, scores=ultimate_scores)


# # --- 5. الدالة الرئيسية للتنفيذ (محسّنة) ---
# async def main():
#     print("--- 🚀 بدء تهيئة مختبر المواجهة النهائية المحسّن (v2.0) 🚀 ---")
#     try:
#         embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
#         vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
#         # **التحسين: تهيئة نموذج CrossEncoder القوي**
#         reranker = CrossEncoder(RERANK_MODEL)

#         all_docs = list(vector_store.docstore._dict.values())
#         all_tenant_docs = {}
#         for doc in all_docs:
#             tenant_id = doc.metadata.get("tenant_id")
#             if tenant_id:
#                 if tenant_id not in all_tenant_docs:
#                     all_tenant_docs[tenant_id] = []
#                 all_tenant_docs[tenant_id].append(doc)

#         # **التحسين: تهيئة المسترجعات مسبقًا وتخزينها في ذاكرة مؤقتة**
#         print("--- ⏳ تهيئة المسترجعات مسبقًا لكل عميل... ---")
#         retrievers_cache = {}
#         for tenant_id, tenant_docs in all_tenant_docs.items():
#             print(f"   -> تهيئة للعميل: {tenant_id}")
#             faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
#             bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=TOP_K)
#             # **التحسين: تعديل الأوزان**
#             ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
#             store = InMemoryStore()
#             parent_document_retriever = ParentDocumentRetriever(vectorstore=vector_store, docstore=store, child_splitter=RecursiveCharacterTextSplitter(chunk_size=400))
#             parent_document_retriever.add_documents(tenant_docs, ids=None)

#             retrievers_cache[tenant_id] = {
#                 'faiss': faiss_retriever,
#                 'bm25': bm25_retriever,
#                 'ensemble': ensemble_retriever,
#                 'parent': parent_document_retriever
#             }

#         print("--- ✅ البيئة جاهزة. ---")
#     except Exception as e:
#         print(f"❌ فشل فادح في تهيئة البيئة: {e}")
#         return

#     # --- تعريف حالات الاختبار الجديدة والمتنوعة ---
#     test_cases = [
#         {
#             "tenant_id": "school_beta",
#             "question": "ما هي مكتبة TensorFlow؟"
#         },
#         {
#             "tenant_id": "school_beta",
#             "question": "قارن بين الطبقة التلافيفية والطبقة الكثيفة."
#         },
#         {
#             "tenant_id": "sys",
#             "question": "ماذا يحدث بعد سداد الفاتورة المبدئية في رحلة الحصول على الاعتماد؟"
#         },
#         {
#             "tenant_id": "school_beta",
#             "question": "كيف يمكن مواجهة مشكلة تلاشي مشتقة الخطأ (Vanishing Gradient)؟"
#         },
#         {
#             "tenant_id": "university_alpha",
#             "question": "ما هي الفائدة الاقتصادية لتطبيق Plant Care للمزارعين، وما هي حدوده الوظيفية؟"
#         }
#     ]

#     # --- تشغيل جميع حالات الاختبار ---
#     for case in test_cases:
#         await run_final_showdown_lab(
#             question=case["question"],
#             tenant_id=case["tenant_id"],
#             vector_store=vector_store,
#             reranker=reranker,
#             retrievers_cache=retrievers_cache
#         )

# if __name__ == "__main__":
#     # ملاحظة: قد تحتاج إلى تثبيت sentence-transformers
#     # pip install -U sentence-transformers
#     asyncio.run(main())
# final_retrieval_lab.py - الاختبار النهائي لاختيار أفضل استراتيجية استرجاع

import os
import asyncio
import time
from typing import List, Dict
from dotenv import load_dotenv

# --- 1. استيراد المكونات ---
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers.cross_encoder import CrossEncoder

# --- 2. الإعدادات النهائية ---
load_dotenv()

# القرار الهندسي: استخدام النموذج الذي أثبت تفوقه
FINAL_EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'} # استخدم 'cuda' إذا كان لديك GPU
)
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# مسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))
TOP_K = 7

# --- 3. دالة مساعدة لعرض النتائج (لا تغيير) ---
def print_results(docs: List[Document], title: str, duration: float, scores: List[float] = None):
    print("\n" + "="*80)
    print(f"🔬 نتائج طريقة: {title}")
    print(f"⏱️ زمن الاسترجاع: {duration:.4f} ثانية")
    print(f"📄 عدد النتائج: {len(docs)}")
    print("="*80)
    if not docs:
        print("   -> لم يتم العثور على نتائج.")
        return
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get('source', 'N/A'))
        content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:110]
        score_info = f"[الدرجة: {scores[i]:.4f}]" if scores is not None else ""
        print(f"   {i+1}. {score_info} [{source}] -> \"{content_preview}...\"")
    print("-" * 80)

# --- 4. المختبر الرئيسي ---
async def run_retrieval_test(
    question: str,
    tenant_id: str,
    retrievers_cache: Dict[str, Dict],
    reranker: CrossEncoder
):
    print("\n" + "#"*30 + f" بدء الاختبار للعميل: '{tenant_id}' | السؤال: '{question}' " + "#"*30)

    tenant_retrievers = retrievers_cache.get(tenant_id)
    if not tenant_retrievers:
        print(f"❌ خطأ: لا توجد مسترجعات مهيأة للعميل '{tenant_id}'.")
        return

    # استرجاع المسترجعات المهيأة مسبقًا
    faiss_retriever = tenant_retrievers['faiss']
    bm25_retriever = tenant_retrievers['bm25']
    ensemble_retriever = tenant_retrievers['ensemble']
    parent_document_retriever = tenant_retrievers['parent']

    # --- التقنية 1: Vector Search ---
    start_time = time.time()
    vector_docs = await faiss_retriever.ainvoke(question)
    duration = time.time() - start_time
    print_results(vector_docs, "1. البحث الدلالي (Vector Search)", duration)

    # --- التقنية 2: Hybrid Search ---
    start_time = time.time()
    hybrid_docs = await ensemble_retriever.ainvoke(question)
    duration = time.time() - start_time
    print_results(hybrid_docs, "2. البحث الهجين (Hybrid - 70/30)", duration)

    # --- التقنية 3: Parent Document Retriever ---
    start_time = time.time()
    parent_docs = await asyncio.to_thread(parent_document_retriever.invoke, question)
    duration = time.time() - start_time
    print_results(parent_docs, "3. مسترجع المستند الأصل (Parent)", duration)

    # --- التقنية 4: المسترجع الشامل (Hybrid + Parent + Reranker) ---
    combined_docs = list({doc.page_content: doc for doc in reversed(hybrid_docs + parent_docs)}.values())[::-1]
    if combined_docs:
        start_time = time.time()
        passages = [[question, doc.page_content] for doc in combined_docs]
        scores = reranker.predict(passages)
        rerank_duration = time.time() - start_time
        
        reranked_results = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in reranked_results][:TOP_K]
        final_scores = [score for doc, score in reranked_results][:TOP_K]
        print_results(final_docs, "4. المسترجع الشامل (Hybrid + Parent + Reranker)", rerank_duration, scores=final_scores)

# --- 5. الدالة الرئيسية للتنفيذ ---
async def main():
    print("--- 🚀 بدء مختبر الاسترجاع النهائي (v4.0) 🚀 ---")
    try:
        # الخطوة 1: تحميل قاعدة المعرفة باستخدام نموذج التضمين الصحيح
        print(f"[*] تحميل قاعدة المعرفة من: '{UNIFIED_DB_PATH}'")
        vector_store = FAISS.load_local(
            UNIFIED_DB_PATH, 
            embeddings=FINAL_EMBEDDING_MODEL, 
            allow_dangerous_deserialization=True
        )
        print("[*] تم تحميل قاعدة المعرفة بنجاح.")

        # الخطوة 2: تهيئة Reranker
        reranker = CrossEncoder(RERANK_MODEL)
        
        # الخطوة 3: جمع كل المقاطع وتوزيعها حسب العميل
        # .docstore._dict هي طريقة داخلية، من الأفضل تجنبها. سنستخدم طريقة أكثر عمومية.
        # نفترض أن الفهرس يحتوي على أرقام من 0 إلى N-1
        total_docs = len(vector_store.index_to_docstore_id)
        all_chunks = [vector_store.docstore.search(vector_store.index_to_docstore_id[i]) for i in range(total_docs)]
        
        all_tenant_chunks = {}
        for chunk in all_chunks:
            tenant_id = chunk.metadata.get("tenant_id")
            if tenant_id:
                if tenant_id not in all_tenant_chunks:
                    all_tenant_chunks[tenant_id] = []
                all_tenant_chunks[tenant_id].append(chunk)

        # الخطوة 4: تهيئة المسترجعات مسبقًا لكل عميل
        print("--- ⏳ تهيئة المسترجعات مسبقًا لكل عميل... ---")
        retrievers_cache = {}
        for tenant_id, tenant_chunks in all_tenant_chunks.items():
            print(f"   -> تهيئة للعميل: {tenant_id}")
            faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
            bm25_retriever = BM25Retriever.from_documents(tenant_chunks, k=TOP_K)
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
            
            # Parent retriever يحتاج إلى المقاطع الأصلية، والتي لا نملكها مباشرة هنا.
            # كحل بديل للاختبار، سنبني Docstore مؤقت من المقاطع المتاحة.
            # في نظام إنتاجي، قد تحتاج إلى تخزين المستندات الأصلية بشكل منفصل.
            store = InMemoryStore()
            store.mset([(str(i), doc) for i, doc in enumerate(tenant_chunks)])
            
            parent_document_retriever = ParentDocumentRetriever(
                vectorstore=vector_store, 
                docstore=store, 
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250),
            )
            
            retrievers_cache[tenant_id] = {
                'faiss': faiss_retriever,
                'bm25': bm25_retriever,
                'ensemble': ensemble_retriever,
                'parent': parent_document_retriever
            }

        print("--- ✅ البيئة جاهزة للاختبار. ---")
    except Exception as e:
        print(f"❌ فشل فادح في تهيئة البيئة: {e}")
        return

    # --- تعريف حالات الاختبار المعيارية ---
    test_cases = [
        {"tenant_id": "school_beta", "question": "ما هي مكتبة TensorFlow؟"},
        {"tenant_id": "school_beta", "question": "قارن بين الطبقة التلافيفية والطبقة الكثيفة."},
        {"tenant_id": "sys", "question": "ماذا يحدث بعد سداد الفاتورة المبدئية في رحلة الحصول على الاعتماد؟"},
        {"tenant_id": "school_beta", "question": "كيف يمكن مواجهة مشكلة تلاشي مشتقة الخطأ (Vanishing Gradient)؟"},
        {"tenant_id": "university_alpha", "question": "ما هي الفائدة الاقتصادية لتطبيق Plant Care للمزارعين، وما هي حدوده الوظيفية؟"},
        {"tenant_id": "un", "question": "ماذا يحدث بعد تقديم العطاء وقبل إرساء العقد؟"}
    ]

    # --- تشغيل جميع حالات الاختبار ---
    for case in test_cases:
        await run_retrieval_test(
            question=case["question"],
            tenant_id=case["tenant_id"],
            retrievers_cache=retrievers_cache,
            reranker=reranker
        )

    print("\n--- 🎉 انتهى الاختبار النهائي. ---")

if __name__ == "__main__":
    asyncio.run(main())

