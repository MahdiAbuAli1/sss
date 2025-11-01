# interactive_hybrid_tester.py

import asyncio
import os
from typing import List 
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document

# --- الإعدادات (نفس الإعدادات السابقة) ---
# ... (انسخ الإعدادات من الكود السابق) ...
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")


# --- دالة جديدة لتحميل كل المستندات الخام (نحتاجها لـ BM25) ---
def load_all_docs_from_faiss(vector_store: FAISS) -> List[Document]:
    # FAISS يخزن المستندات الأصلية. يمكننا استخراجها.
    # docstore.items() يرجع (id, Document)
    return list(vector_store.docstore._dict.values())


async def interactive_hybrid_search():
    """
    دالة لإجراء اختبار تفاعلي للبحث الهجين (الدلالي + الكلمات المفتاحية).
    """
    print("--- بدء تهيئة مكونات البحث الهجين ---")
    
    try:
        # 1. تهيئة المكونات الدلالية (FAISS)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        if not os.path.isdir(UNIFIED_DB_PATH):
            print(f"❌ خطأ: قاعدة البيانات المتجهة غير موجودة.")
            return
        faiss_vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={'k': 4})
        print("✅ تم تحميل المسترجع الدلالي (FAISS).")

        # 2. تهيئة مكونات البحث بالكلمات المفتاحية (BM25)
        print("🔧 جاري بناء مسترجع الكلمات المفتاحية (BM25)...")
        # BM25 يحتاج إلى قائمة المستندات الأصلية لبناء فهرسه
        all_docs = load_all_docs_from_faiss(faiss_vector_store)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 4 # حدد عدد النتائج التي تريدها منه
        print("✅ تم بناء المسترجع (BM25).")

        # 3. إنشاء المسترجع الهجين (EnsembleRetriever)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # إعطاء وزن متساوٍ لكلا الطريقتين
        )
        print("🚀 النظام الهجين جاهز للاختبار!")

    except Exception as e:
        print(f"❌ فشل فادح أثناء التهيئة: {e}")
        return

    # 4. بدء الحلقة التفاعلية
    while True:
        print("\n" + "="*50)
        question = input("🖋️ أدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")
        print("="*50)

        if question.lower().strip() in ['خروج', 'exit', 'quit']:
            print("👋 وداعاً!")
            break
        
        if not question.strip():
            continue

        print(f"\n🔍 جاري البحث الهجين عن: '{question}'...")
        
        # 5. استدعاء المسترجع الهجين
        try:
            # .ainvoke هي النسخة غير المتزامنة من .invoke
            retrieved_docs = await ensemble_retriever.ainvoke(question)
            
            if not retrieved_docs:
                print("\n--- لم يتم العثور على نتائج ---")
                continue

            print(f"\n--- النتائج الهجينة (عدد: {len(retrieved_docs)}) ---")
            
            for i, doc in enumerate(retrieved_docs):
                print(f"\n📄 المستند رقم {i+1}:")
                content_preview = ' '.join(doc.page_content.split())
                print(f"   المحتوى: {content_preview[:350]}...")
                if doc.metadata:
                    print(f"   بيانات وصفية (Metadata): {doc.metadata}")

        except Exception as e:
            print(f"❌ حدث خطأ أثناء البحث: {e}")
            
    print("\n--- انتهى الاختبار ---")

if __name__ == "__main__":
    asyncio.run(interactive_hybrid_search())
