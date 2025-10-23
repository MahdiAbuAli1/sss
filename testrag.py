# testrag.py
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "3_shared_resources/vector_db"))

if not EMBEDDING_MODEL_NAME:
    raise ValueError(" خطأ: متغير EMBEDDING_MODEL_NAME فارغ! تحقق من ملف .env")

def test_tenant_retrieval(tenant_id: str, question: str, k: int = 4):
    print(f"🟢 اختبار استرجاع البيانات للـ tenant_id='{tenant_id}' مع السؤال: '{question}'")
    
    # تحميل نموذج التضمين
    embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # تحميل قاعدة FAISS
    if not os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        raise FileNotFoundError(f"❌ قاعدة المعرفة غير موجودة في {VECTOR_DB_PATH}")
    
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings_model, allow_dangerous_deserialization=True)

    # تحويل السؤال إلى متجه
    question_vector = embeddings_model.embed_query(question)

    # استرجاع المستندات باستخدام المتجه
    docs = vector_store.similarity_search_by_vector(question_vector, k=k)

    print(f"==== المستندات المسترجعة ====")
    for i, doc in enumerate(docs):
        print(f"--- Document/Chunk {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content[:500] + ('...' if len(doc.page_content) > 500 else ''))  # نعرض أول 500 حرف فقط
        print("\n")

if __name__ == "__main__":
    test_tenant_retrieval("university_alpha", "ما هو النظام المقترح؟", k=4)
