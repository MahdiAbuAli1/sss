# project_core/processing/run_pipeline.py

import os
import logging
import json
from datetime import datetime

# تجاهل تحذيرات الإهمال لجعل المخرجات أنظف
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore

from project_core.core.config import (
    DATA_SOURCES_DIR, VECTORSTORE_PATH, DOCSTORE_PATH, LOGS_DIR, BASE_DIR, get_embeddings_model,
)
from project_core.processing.pipeline import process_document_elements

# ==============================================================================
# 0. إعداد نظام التسجيل (Logging)
# ==============================================================================
def setup_logging():
    """ إعداد نظام التسجيل لحفظ المخرجات في ملف وعرضها على الشاشة. """
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    log_filename = datetime.now().strftime(f"pipeline_run_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(LOGS_DIR, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("run_pipeline")
    logger.info(f"سيتم حفظ سجلات هذه العملية في الملف: {log_filepath}")
    return logger

logger = setup_logging()

# ==============================================================================
# 1. إدارة سجل الملفات المعالجة
# ==============================================================================
PROCESSED_FILES_LOG = os.path.join(BASE_DIR, "processed_files.json")

def load_processed_files_log():
    """ تحميل سجل الملفات التي تمت معالجتها من ملف JSON. """
    if os.path.exists(PROCESSED_FILES_LOG):
        try:
            with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("ملف سجل المعالجة تالف. سيتم البدء من جديد.")
            return {}
    return {}

def save_processed_files_log(log_data):
    """ حفظ سجل الملفات التي تمت معالجتها في ملف JSON. """
    with open(PROCESSED_FILES_LOG, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

# ==============================================================================
# 2. الدالة الرئيسية لتشغيل خط الأنابيب الذكي
# ==============================================================================
def main():
    """
    الدالة الرئيسية التي تقوم بالمعالجة الذكية والتزايدية للملفات.
    """
    processed_log = load_processed_files_log()
    files_to_process = []

    logger.info("="*50 + "\n🚀 بدء عملية المعالجة الذكية (التزايدية)...\n" + "="*50)

    if not os.path.exists(DATA_SOURCES_DIR):
        logger.error(f"مجلد المصادر '{DATA_SOURCES_DIR}' غير موجود. لا يمكن المتابعة.")
        return

    # --- الخطوة 1: تحديد الملفات التي تحتاج إلى معالجة (جديدة أو محدثة) ---
    logger.info("--- المرحلة 1: التحقق من الملفات ---")
    for tenant_id in os.listdir(DATA_SOURCES_DIR):
        tenant_path = os.path.join(DATA_SOURCES_DIR, tenant_id)
        if os.path.isdir(tenant_path):
            for file_name in os.listdir(tenant_path):
                file_path = os.path.join(tenant_path, file_name)
                if not os.path.isfile(file_path): continue
                
                try:
                    file_mod_time = os.path.getmtime(file_path)
                    if file_path not in processed_log or processed_log[file_path] < file_mod_time:
                        logger.info(f"✔️ [للمعالجة] ملف جديد أو محدث: {file_name}")
                        files_to_process.append((file_path, tenant_id, file_mod_time))
                    else:
                        logger.info(f"⚪️ [تجاهل] ملف لم يتغير: {file_name}")
                except FileNotFoundError:
                    logger.warning(f"تم العثور على ملف في السجل لم يعد موجودًا: {file_path}. سيتم تجاهله.")
                    continue

    if not files_to_process:
        logger.info("\n✅ كل الملفات محدّثة. لا يوجد شيء للمعالجة. انتهى.")
        return

    # --- الخطوة 2: معالجة الملفات المحددة فقط ---
    logger.info("\n--- المرحلة 2: معالجة الملفات المحددة ---")
    all_docs, all_ids, all_contents = [], [], []
    
    for file_path, tenant_id, file_mod_time in files_to_process:
        docs, ids, contents = process_document_elements(file_path, tenant_id)
        if docs:
            all_docs.extend(docs)
            all_ids.extend(ids)
            all_contents.extend(contents)
            # تحديث سجل المعالجة فقط عند النجاح
            processed_log[file_path] = file_mod_time
    
    # --- الخطوة 3: تخزين البيانات الجديدة فقط ---
    if all_docs:
        logger.info("\n--- المرحلة 3: تخزين البيانات الجديدة في قواعد البيانات ---")
        try:
            vectorstore = Chroma(collection_name="rag-chroma", embedding_function=get_embeddings_model(), persist_directory=VECTORSTORE_PATH)
            doc_store = LocalFileStore(DOCSTORE_PATH)
            
            vectorstore.add_documents(all_docs)
            doc_store.mset(list(zip(all_ids, all_contents)))
            
            save_processed_files_log(processed_log) # حفظ السجل المحدث
            logger.info("\n🎉 اكتملت عملية التخزين بنجاح!")
            logger.info(f"  - تم تحديث قاعدة المتجهات في: '{VECTORSTORE_PATH}'")
            logger.info(f"  - تم تحديث مخزن المستندات في: '{DOCSTORE_PATH}'")
        except Exception as e:
            logger.error(f"حدث خطأ فادح أثناء تخزين البيانات: {e}")
    else:
        logger.warning("لم يتم العثور على عناصر جديدة للتخزين بعد المعالجة.")

# --- نقطة انطلاق البرنامج ---
if __name__ == "__main__":
    main()
