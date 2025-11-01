# # المسار: 2_central_api_service/agent_app/advanced_tester.py
# # --- النسخة فائقة السرعة (مع محاكاة إعادة صياغة السؤال) ---

# import asyncio
# import os
# from typing import List, Tuple
# from dotenv import load_dotenv

# # --- استيراد مكونات LangChain (لا تغيير) ---
# from langchain_core.documents import Document
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import BM25Retriever, EnsembleRetriever

# # --- الإعدادات الأساسية (لا تغيير) ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- دوال مساعدة (لا تغيير) ---
# def _load_all_docs_from_faiss(vector_store: FAISS) -> List[Document]:
#     return list(vector_store.docstore._dict.values())

# def print_results(docs: List[Document], title: str):
#     print(f"\n--- {title} ---")
#     if not docs:
#         print("   -> لم يتم العثور على نتائج.")
#         return
#     print(f"   -> عدد النتائج: {len(docs)}")
#     for i, doc in enumerate(docs):
#         source = doc.metadata.get('source', 'غير معروف').split('\\')[-1]
#         tenant = doc.metadata.get('tenant_id', 'N/A')
#         content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:100]
#         print(f"   {i+1}. [العميل: {tenant}, المصدر: {source}] -> \"{content_preview}...\"")
#     print("-" * (len(title) + 6))

# # --- تهيئة البيئة (تعديل بسيط لإزالة LLM غير الضروري الآن) ---
# async def setup_environment() -> FAISS:
#     print("--- 🔬 بدء تهيئة بيئة الاختبار (للاسترجاع فقط) 🔬 ---")
#     embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=os.getenv("OLLAMA_HOST"))
#     if not os.path.isdir(UNIFIED_DB_PATH):
#         raise FileNotFoundError("قاعدة البيانات المتجهة غير موجودة.")
#     faiss_vector_store = await asyncio.to_thread(
#         FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#     )
#     print("--- ✨ بيئة الاختبار جاهزة (FAISS store) ✨ ---\n")
#     return faiss_vector_store

# # --- تجربة المرحلة الثانية (بشكل محاكى وسريع) ---

# async def experiment_2_1_simulated_query_rewriting(vector_store: FAISS, question: str, tenant_id: str):
#     """
#     التجربة 2.1 (محاكاة): اختبار تأثير إعادة صياغة السؤال بدون انتظار LLM.
#     """
#     print("\n" + "="*60)
#     print(f"🔬 التجربة 2.1 (محاكاة): السؤال: '{question}' للعميل '{tenant_id}'")
#     print("="*60)

#     # بناء المسترجع الهجين المفلتر (نفس المنطق السابق)
#     all_docs = _load_all_docs_from_faiss(vector_store)
#     tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
#     if not tenant_docs:
#         print(f"❌ لا توجد مستندات للعميل '{tenant_id}'.")
#         return
    
#     bm25_retriever = BM25Retriever.from_documents(tenant_docs)
#     bm25_retriever.k = 5
#     faiss_retriever = vector_store.as_retriever(
#         search_kwargs={'k': 5, 'filter': {'tenant_id': tenant_id}}
#     )
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

#     # البحث بالسؤال الأصلي
#     original_docs = await ensemble_retriever.ainvoke(question)
#     print_results(original_docs, "1. النتائج بالسؤال الأصلي (الغامض)")

#     # *** المحاكاة الذكية (الغش) ***
#     # بدلاً من انتظار LLM، سنكتب بأنفسنا السؤال الذي نتوقع أن يولده
#     print("\n🧠 محاكاة لـ LLM: نقوم بإعادة صياغة السؤال يدويًا...")
#     simulated_rewritten_query = "المتطلبات الوظيفية وغير الوظيفية لتطبيق المستخدم والمدير"
#     print(f"✨ السؤال المُعاد صياغته (المحاكى): '{simulated_rewritten_query}'")

#     # البحث بالسؤال المحاكى
#     rewritten_docs = await ensemble_retriever.ainvoke(simulated_rewritten_query)
#     print_results(rewritten_docs, "2. النتائج بالسؤال المُعاد صياغته (المحاكى)")

# # --- الدالة الرئيسية ---

# async def main():
#     vector_store = await setup_environment()
    
#     failed_question = "ما هي مكونات نظام العميل؟"
#     target_tenant = "university_alpha"
    
#     await experiment_2_1_simulated_query_rewriting(vector_store, question=failed_question, tenant_id=target_tenant)

# if __name__ == "__main__":
#     asyncio.run(main())


# المسار: 2_central_api_service/agent_app/advanced_tester.py
# --- النسخة المحدثة مع التجربة 2.2: التوجيه المخصص ---

import asyncio
import os
from typing import List, Tuple
from dotenv import load_dotenv

# --- استيراد مكونات LangChain (لا تغيير) ---
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# --- الإعدادات الأساسية (لا تغيير) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# --- دوال مساعدة (لا تغيير) ---
def _load_all_docs_from_faiss(vector_store: FAISS) -> List[Document]:
    return list(vector_store.docstore._dict.values())

def print_results(docs: List[Document], title: str):
    print(f"\n--- {title} ---")
    if not docs:
        print("   -> لم يتم العثور على نتائج.")
        return
    print(f"   -> عدد النتائج: {len(docs)}")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'غير معروف').split('\\')[-1]
        tenant = doc.metadata.get('tenant_id', 'N/A')
        content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:100]
        print(f"   {i+1}. [العميل: {tenant}, المصدر: {source}] -> \"{content_preview}...\"")
    print("-" * (len(title) + 6))

# --- تهيئة البيئة (لا تغيير) ---
async def setup_environment() -> FAISS:
    print("--- 🔬 بدء تهيئة بيئة الاختبار (للاسترجاع فقط) 🔬 ---")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=os.getenv("OLLAMA_HOST"))
    if not os.path.isdir(UNIFIED_DB_PATH):
        raise FileNotFoundError("قاعدة البيانات المتجهة غير موجودة.")
    faiss_vector_store = await asyncio.to_thread(
        FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print("--- ✨ بيئة الاختبار جاهزة (FAISS store) ✨ ---\n")
    return faiss_vector_store

# --- تجربة المرحلة 2.2: التوجيه المخصص (محاكاة) ---

async def experiment_2_2_context_aware_rewriting(vector_store: FAISS, question: str, tenant_id: str):
    """
    التجربة 2.2 (محاكاة): اختبار تأثير إعطاء النموذج "ملف شخصي للنظام" قبل إعادة الصياغة.
    """
    print("\n" + "="*60)
    print(f"🔬 التجربة 2.2 (محاكاة): السؤال: '{question}' للعميل '{tenant_id}'")
    print("="*60)

    # --- محاكاة "الملف الشخصي للنظام" ---
    # في نظام حقيقي، سيتم تحميل هذه البيانات من ملف config.json أو قاعدة بيانات
    system_profiles = {
        "sys": {
            "name": "نظام إدارة طلبات الاعتماد",
            "description": "نظام لتتبع مراحل الحصول على الاعتماد من التقديم حتى إصدار الشهادة.",
            "keywords": ["طلب اعتماد", "قوائم التحقق", "دراسة مكتبية", "زيارة ميدانية", "إجراءات تصحيحية"]
        },
        "university_alpha": {
            "name": "تطبيق Plant Care",
            "description": "تطبيق ذكي لمساعدة المزارعين في التعرف على الآفات الزراعية.",
            "keywords": ["متطلبات وظيفية", "حالات استخدام", "تصميم النظام", "مخطط علاقات", "plant care"]
        }
    }
    
    profile = system_profiles.get(tenant_id)
    if not profile:
        print(f"❌ لا يوجد ملف شخصي للعميل '{tenant_id}'.")
        return

    print(f"👤 تم العثور على ملف شخصي لـ '{profile['name']}'")

    # بناء المسترجع الهجين المفلتر (نفس المنطق السابق)
    all_docs = _load_all_docs_from_faiss(vector_store)
    tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
    bm25_retriever = BM25Retriever.from_documents(tenant_docs)
    bm25_retriever.k = 5
    faiss_retriever = vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'tenant_id': tenant_id}})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    # *** المحاكاة الذكية (باستخدام الملف الشخصي) ***
    print("\n🧠 محاكاة لـ LLM (مع سياق النظام): نقوم بصياغة سؤال أفضل يدويًا...")
    
    # مثال على كيفية استخدام الملف الشخصي لتوليد سؤال أفضل
    # سنختار كلمة مفتاحية من الملف الشخصي ونبحث عنها
    simulated_rewritten_query = f"خطوات {profile['keywords'][0]}"
    
    print(f"✨ السؤال الأصلي: '{question}'")
    print(f"✨ السؤال المُعاد صياغته (المحاكى): '{simulated_rewritten_query}'")

    # البحث بالسؤال المحاكى
    rewritten_docs = await ensemble_retriever.ainvoke(simulated_rewritten_query)
    print_results(rewritten_docs, f"النتائج باستخدام السؤال المحاكى والمخصص لنظام '{profile['name']}'")

# --- الدالة الرئيسية ---

async def main():
    vector_store = await setup_environment()
    
    # --- تشغيل التجربة 2.2 على نظام الاعتماد ---
    # سؤال عام جدًا
    generic_question = "كيف أبدأ؟"
    target_tenant_sys = "sys"
    
    await experiment_2_2_context_aware_rewriting(vector_store, question=generic_question, tenant_id=target_tenant_sys)

if __name__ == "__main__":
    asyncio.run(main())
