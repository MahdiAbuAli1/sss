import asyncio
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#المسارات والاعدادت
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

async def interactive_semantic_search():
    """
    دالة لإجراء اختبار تفاعلي للبحث الدلالي عبر الطرفية.
    """
    print("--- بدء تهيئة مكونات البحث الدلالي ---")
    
    try:
        # 1. تهيئة نموذج التضمين وقاعدة البيانات المتجهة
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        
        if not os.path.isdir(UNIFIED_DB_PATH):
            print(f"❌ خطأ: قاعدة البيانات المتجهة غير موجودة في المسار: {UNIFIED_DB_PATH}")
            return

        vector_store = FAISS.load_local(
            UNIFIED_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ تم تحميل قاعدة البيانات والمسترجع بنجاح. النظام جاهز للاختبار.")
        
        # 2. إنشاء المسترجع (Retriever)
        retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    except Exception as e:
        print(f"❌ فشل فادح أثناء التهيئة: {e}")
        return

    # 3. بدء الحلقة التفاعلية لجعل المستخدم يدخل الأسئلة
    while True:
        print("\n" + "="*50)
        # استخدام input() لجعل المستخدم يكتب السؤال
        question = input("🖋️ أدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")
        print("="*50)

        if question.lower().strip() in ['خروج', 'exit', 'quit']:
            print("👋 وداعاً!")
            break
        
        if not question.strip():
            print("لم تدخل سؤالاً. يرجى المحاولة مرة أخرى.")
            continue

        print(f"\n🔍 جاري البحث عن مستندات ذات صلة بالسؤال: '{question}'...")
        
        # 4. استدعاء المسترجع مباشرةً
        try:
            retrieved_docs = await retriever.ainvoke(question)
            
            if not retrieved_docs:
                print("\n--- لم يتم العثور على نتائج ---")
                continue

            print(f"\n--- النتائج المسترجعة (عدد: {len(retrieved_docs)}) ---")
            
            # 5. طباعة محتوى المستندات التي تم العثور عليها
            for i, doc in enumerate(retrieved_docs):
                print(f"\n📄 المستند رقم {i+1}:")
                # طباعة المحتوى مع إزالة المسافات الزائدة والأسطر الجديدة لتسهيل القراءة
                content_preview = ' '.join(doc.page_content.split())
                print(f"   المحتوى: {content_preview[:350]}...")
                if doc.metadata:
                    print(f"   بيانات وصفية (Metadata): {doc.metadata}")

        except Exception as e:
            print(f"❌ حدث خطأ أثناء البحث: {e}")
            
    print("\n--- انتهى الاختبار التفاعلي ---")

# لتشغيل السكربت
if __name__ == "__main__":
    # لتجنب مشاكل في Windows مع asyncio و input
    # قد تحتاج إلى تثبيت aiohttp_jinja2 و aiohttp إذا واجهت مشاكل
    try:
        asyncio.run(interactive_semantic_search( ))
    except KeyboardInterrupt:
        print("\nتم إيقاف البرنامج.")
