# 1_knowledge_pipeline/vector_store_manager.py
# ------pip install python-dotenv filelock
# إذا كنت ستستخدم HuggingFace embeddings
#pip install sentence-transformers
#
# -----------------------------------------------------------------------
# مدير قاعدة بيانات المتجهات الاحترافي.
# - يستخدم متغيرات البيئة للإعدادات.
# - يدعم آلية قفل بسيطة لمنع الكتابة المتزامنة.
# - يوفر تسجيلًا واضحًا للعمليات والأخطاء.
# -----------------------------------------------------------------------------
import os
import os
import argparse
from dotenv import load_dotenv

# هذا هو أهم سطر في الملف بأكمله، يجب أن يكون الأول
load_dotenv()

# --- الخطوة 1: استيراد وحدات المشروع ---
# الآن بعد أن تم تحميل متغيرات البيئة، يمكننا استيراد وحداتنا بأمان.
from loaders import load_documents
from cleaners import clean_documents
from splitters import split_documents
from vector_store_manager import add_to_vector_store
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from filelock import FileLock, Timeout

# --- إعداد التسجيل (Logging) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- تعريف الثوابت ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))
LOCK_FILE_PATH = os.path.join(VECTOR_DB_DIR, "faiss.lock")

# --- قراءة الإعدادات من متغيرات البيئة (التي تم تحميلها بواسطة main_builder.py) ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default_model_name")

def get_embeddings_model():
    """ يقوم بتهيئة وإرجاع نموذج التضمين. """
    logging.info(f"إعداد نموذج التضمين بالاسم: '{EMBEDDING_MODEL_NAME}'...")
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

def add_to_vector_store(chunks: List[Document]):
    """
    يضيف قائمة من القطع (Chunks) إلى قاعدة بيانات FAISS بشكل آمن.
    يستخدم قفل ملفات لمنع تضارب عمليات الكتابة.
    """
    if not chunks:
        logging.warning("لا توجد قطع لإضافتها إلى قاعدة المعرفة. تم التخطي.")
        return

    logging.info(f"المرحلة 5: سيتم إضافة {len(chunks)} قطعة إلى قاعدة المعرفة...")
    
    # تأكد من وجود مجلد قاعدة البيانات
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    lock = FileLock(LOCK_FILE_PATH)

    try:
        with lock.acquire(timeout=60): # انتظر حتى 60 ثانية للحصول على القفل
            logging.info("تم الحصول على قفل الكتابة لقاعدة المعرفة.")
            
            embeddings_model = get_embeddings_model()
            
            db_exists = os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 1

            if not db_exists:
                logging.info(f"قاعدة المعرفة في '{VECTOR_DB_DIR}' غير موجودة. سيتم إنشاؤها...")
                vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
                vector_store.save_local(VECTOR_DB_DIR)
                logging.info("تم إنشاء وحفظ قاعدة المعرفة بنجاح.")
            else:
                logging.info(f"تحميل قاعدة المعرفة الموجودة من '{VECTOR_DB_DIR}'...")
                vector_store = FAISS.load_local(
                    VECTOR_DB_DIR, 
                    embeddings=embeddings_model, 
                    allow_dangerous_deserialization=True
                )
                logging.info("جارٍ دمج القطع الجديدة...")
                vector_store.add_documents(documents=chunks)
                vector_store.save_local(VECTOR_DB_DIR)
                logging.info("تم دمج القطع الجديدة وحفظ قاعدة المعرفة المحدثة بنجاح.")

    except Timeout:
        logging.error("فشل الحصول على قفل الكتابة. قد تكون هناك عملية أخرى تستخدم قاعدة المعرفة.")
    except Exception as e:
        logging.critical(f"حدث خطأ فادح أثناء التعامل مع قاعدة المعرفة: {e}", exc_info=True)
    finally:
        logging.info("تم تحرير قفل الكتابة.")
