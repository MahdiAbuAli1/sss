# p_v2.py - اختبار الاسترجاع بدون عتبة درجة
import asyncio
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# --- الإعدادات ---
load_dotenv()
# استخدم النموذج الذي تثق به
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b") 
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

async def main():
    print(f"--- 🔬 بدء اختبار الاسترجاع (بدون عتبة) للنموذج: {EMBEDDING_MODEL} ---")
    
    # --- 1. تهيئة البيئة ---
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ تم تحميل قاعدة البيانات FAISS بنجاح.")
    except Exception as e:
        print(f"فشل في تهيئة البيئة: {e}")
        return

    # --- 2. إعداد المسترجع الدلالي (بدون عتبة) ---
    # [تصحيح] تم تغيير search_type إلى "similarity" لإرجاع أفضل K نتائج دائمًا.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5, 'filter': {'tenant_id': 'sys'}}
    )

    # --- 3. جملة البحث ---
    query = "دليل استخدام نظام إدارة طلبات الاعتماد"
    print(f"\n🔍 البحث عن الجملة: '{query}'")

    # --- 4. تنفيذ البحث ---
    try:
        # [تصحيح] سنستخدم الآن `get_relevant_documents_with_score` إذا كان متاحًا، أو `ainvoke`
        # الطريقة الأكثر موثوقية للحصول على الدرجات هي `similarity_search_with_score`
        results_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score,
            query,
            k=5,
            filter={'tenant_id': 'sys'}
        )
        
        print(f"\n--- 📊 النتائج (عدد: {len(results_with_scores)}) ---")
        if not results_with_scores:
            print("   -> ❌ فشل غريب: لم يتم العثور على أي مستندات على الإطلاق.")
        else:
            print("   -> ✅ نجاح! تم استرجاع المستندات. انظر إلى الدرجات:")
            for i, (doc, score) in enumerate(results_with_scores):
                content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:100]
                # ملاحظة: FAISS يُرجع المسافة (distance)، وليس التشابه (similarity).
                # درجة 0 هي تطابق تام. درجة أعلى تعني تشابه أقل.
                print(f"   {i+1}. [المسافة: {score:.4f}] -> \"{content_preview}...\"")
            
            print("\n--- 🕵️‍♂️ التحليل ---")
            print("إذا كانت 'المسافة' لأفضل نتيجة قريبة من الصفر (مثلاً أقل من 1.0)، فهذا يعني أن النموذج يعمل بشكل ممتاز.")
            print("المشكلة كانت فقط في عتبة الدرجة (score_threshold) التي كانت متشددة جدًا.")

    except Exception as e:
        print(f"❌ حدث خطأ أثناء البحث: {e}")

if __name__ == "__main__":
    asyncio.run(main())
