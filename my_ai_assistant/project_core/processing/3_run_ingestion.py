# # project_core/processing/3_run_ingestion.py

# import os
# import logging
# import json
# from datetime import datetime
# from tqdm import tqdm

# # --- استيراد المكتبات الضرورية ---
# from langchain_community.vectorstores.chroma import Chroma
# from langchain.storage import LocalFileStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document

# # --- استيراد الوحدات المخصصة ---
# from project_core.core.config import (
#     get_embeddings_model,
#     # تم حذف ENRICHED_DIR من هنا
#     VECTORSTORE_PATH,
#     DOCSTORE_PATH,
#     COLLECTION_NAME,
#     BASE_DIR  # سنستخدم هذا لتعريف المجلدات
# )
# from project_core.processing.utils import load_processed_files_log, save_processed_files_log

# # --- إعداد نظام التسجيل ---
# LOGS_DIR = os.path.join(BASE_DIR, "logs")
# if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
# log_filename = datetime.now().strftime(f"ingestion_run_%Y-%m-%d_%H-%M-%S.log")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.FileHandler(os.path.join(LOGS_DIR, log_filename), encoding='utf-8'), logging.StreamHandler()])
# logger = logging.getLogger("ingestion_pipeline")

# # --- تعريف مجلد الإدخال (التعديل هنا) ---
# ENRICHED_DIR = os.path.join(BASE_DIR, "enriched_outputs")


# def run_ingestion():
#     """
#     الدالة الرئيسية لتشغيل مرحلة التخزين النهائي للبيانات.
#     """
#     logger.info("="*50 + "\n🚀 بدء المرحلة الثالثة: التخزين الذكي...\n" + "="*50)

#     try:
#         # --- تهيئة نموذج التضمين ---
#         embeddings_model = get_embeddings_model()

#         # --- تهيئة مخزن المستندات (لتخزين المحتوى الأصلي) ---
#         if not os.path.exists(DOCSTORE_PATH): os.makedirs(DOCSTORE_PATH)
#         fs = LocalFileStore(DOCSTORE_PATH)
        
#         # --- تهيئة قاعدة البيانات المتجهة (Vector Store) ---
#         vectorstore = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=embeddings_model,
#             persist_directory=VECTORSTORE_PATH,
#         )
#         logger.info("✅ تم تهيئة قواعد البيانات بنجاح.")

#         # --- تهيئة قاطع النصوص الذكي ---
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             is_separator_regex=False,
#         )
#         logger.info("✅ تم تهيئة قاطع النصوص الذكي.")

#         # --- تحميل سجل الملفات التي تمت معالجتها ---
#         processed_log = load_processed_files_log()
        
#         # --- تحديد الملفات الجديدة التي تحتاج إلى تخزين ---
#         all_files = [f for f in os.listdir(ENRICHED_DIR) if f.endswith(".json")]
#         files_to_process = [f for f in all_files if f not in processed_log]

#         if not files_to_process:
#             logger.warning("🎉 لا توجد ملفات جديدة للتخزين. كل شيء محدّث!")
#             return

#         logger.info(f"🔍 تم العثور على {len(files_to_process)} ملفات جديدة تحتاج إلى تخزين.")

#         all_docs_for_embedding = []
#         all_original_contents = []
#         all_doc_ids = []

#         # --- قراءة ومعالجة الملفات الجديدة ---
#         for filename in tqdm(files_to_process, desc="قراءة الملفات المثرية"):
#             file_path = os.path.join(ENRICHED_DIR, filename)
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 chunks = json.load(f)
            
#             for chunk in chunks:
#                 # إنشاء معرف فريد لكل قطعة أصلية
#                 doc_id = f"{chunk['metadata']['tenant_id']}-{chunk['metadata']['source_file']}-{len(all_doc_ids)}"
                
#                 # --- التعامل مع المحتوى الأصلي ---
#                 original_content = chunk["original_content"]
#                 # التحقق من أن المحتوى بايتات قبل التخزين
#                 if isinstance(original_content, str):
#                     all_original_contents.append(original_content.encode('utf-8'))
#                 else: # يفترض أنه بايتات بالفعل إذا لم يكن نصًا
#                     all_original_contents.append(original_content)
                
#                 all_doc_ids.append(doc_id)

#                 # --- التقطيع الذكي للمحتوى المُثرى ---
#                 enriched_content = chunk["enriched_content"]
                
#                 temp_doc = Document(
#                     page_content=enriched_content,
#                     metadata={
#                         "doc_id": doc_id,
#                         "source_file": chunk["metadata"]["source_file"],
#                         "tenant_id": chunk["metadata"]["tenant_id"],
#                         "type": chunk["type"]
#                     }
#                 )
                
#                 split_docs = text_splitter.split_documents([temp_doc])
#                 all_docs_for_embedding.extend(split_docs)

#         if not all_docs_for_embedding:
#             logger.warning("لم يتم العثور على محتوى للتخزين.")
#             return

#         logger.info(f"💾 بدء تخزين {len(all_original_contents)} قطعة محتوى أصلي و {len(all_docs_for_embedding)} قطعة متجهة...")

#         # --- تخزين المحتوى الأصلي ---
#         try:
#             fs.mset(list(zip(all_doc_ids, all_original_contents)))
#             logger.info("   > ✅ اكتمل تخزين المحتوى الأصلي في مخزن المستندات.")
#         except Exception as store_err:
#             logger.error(f"فشل تخزين المحتوى الأصلي: {store_err}")
#             # يمكنك أن تقرر إيقاف العملية هنا إذا كان هذا خطأً فادحًا
#             return

#         # --- تخزين المتجهات (مع شريط تقدم) ---
#         # إنشاء معرفات فريدة لكل قطعة مقطعة
#         split_ids = [f"{doc.metadata['doc_id']}-{i}" for i, doc in enumerate(all_docs_for_embedding)]
        
#         vectorstore.add_documents(
#             documents=tqdm(all_docs_for_embedding, desc="تضمين وتخزين المتجهات"),
#             ids=split_ids
#         )
#         logger.info("   > ✅ اكتمل تخزين المتجهات في قاعدة البيانات.")

#         # --- تحديث سجل الملفات المعالجة ---
#         processed_log.extend(files_to_process)
#         save_processed_files_log(processed_log)
        
#         logger.info("\n🎉 اكتملت عملية التخزين بنجاح!")

#     except Exception as e:
#         logger.error(f"❌ حدث خطأ فادح أثناء عملية التخزين: {e}")

# if __name__ == "__main__":
#     run_ingestion()
# project_core/processing/3_run_ingestion.py

# import os
# import logging
# import json
# from datetime import datetime
# from tqdm import tqdm
# import time # سنحتاجه للتأخير

# # --- استيراد المكتبات الضرورية ---
# from langchain_community.vectorstores.chroma import Chroma
# from langchain.storage import LocalFileStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document

# # --- استيراد الوحدات المخصصة ---
# from project_core.core.config import (
#     get_embeddings_model,
#     VECTORSTORE_PATH,
#     DOCSTORE_PATH,
#     COLLECTION_NAME,
#     BASE_DIR
# )
# from project_core.processing.utils import load_processed_files_log, save_processed_files_log

# # --- إعداد نظام التسجيل ---
# LOGS_DIR = os.path.join(BASE_DIR, "logs")
# if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
# log_filename = datetime.now().strftime(f"ingestion_run_%Y-%m-%d_%H-%M-%S.log")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.FileHandler(os.path.join(LOGS_DIR, log_filename), encoding='utf-8'), logging.StreamHandler()])
# logger = logging.getLogger("ingestion_pipeline")

# # --- تعريف مجلد الإدخال ---
# ENRICHED_DIR = os.path.join(BASE_DIR, "enriched_outputs")

# def run_ingestion():
#     """
#     الدالة الرئيسية لتشغيل مرحلة التخزين النهائي للبيانات.
#     """
#     logger.info("="*50 + "\n🚀 بدء المرحلة الثالثة: التخزين الذكي (نسخة الدفعات المتحكم بها)...\n" + "="*50)

#     try:
#         embeddings_model = get_embeddings_model()
#         if not os.path.exists(DOCSTORE_PATH): os.makedirs(DOCSTORE_PATH)
#         fs = LocalFileStore(DOCSTORE_PATH)
        
#         vectorstore = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=embeddings_model,
#             persist_directory=VECTORSTORE_PATH,
#         )
#         logger.info("✅ تم تهيئة قواعد البيانات بنجاح.")

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len,
#             is_separator_regex=False,
#         )
#         logger.info("✅ تم تهيئة قاطع النصوص الذكي.")

#         processed_log = load_processed_files_log()
#         all_files = [f for f in os.listdir(ENRICHED_DIR) if f.endswith(".json")]
#         files_to_process = [f for f in all_files if f not in processed_log]

#         if not files_to_process:
#             logger.warning("🎉 لا توجد ملفات جديدة للتخزين. كل شيء محدّث!")
#             return

#         logger.info(f"🔍 تم العثور على {len(files_to_process)} ملفات جديدة تحتاج إلى تخزين.")

#         all_docs_for_embedding = []
#         all_original_contents = []
#         all_doc_ids = []

#         for filename in tqdm(files_to_process, desc="قراءة الملفات المثرية"):
#             file_path = os.path.join(ENRICHED_DIR, filename)
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 chunks = json.load(f)
            
#             for chunk in chunks:
#                 doc_id = f"{chunk['metadata']['tenant_id']}-{chunk['metadata']['source_file']}-{len(all_doc_ids)}"
                
#                 original_content = chunk["original_content"]
#                 encoded_content = original_content.encode('utf-8') if isinstance(original_content, str) else original_content
#                 all_original_contents.append(encoded_content)
#                 all_doc_ids.append(doc_id)

#                 enriched_content = chunk["enriched_content"]
#                 temp_doc = Document(
#                     page_content=enriched_content,
#                     metadata={
#                         "doc_id": doc_id,
#                         "source_file": chunk["metadata"]["source_file"],
#                         "tenant_id": chunk["metadata"]["tenant_id"],
#                         "type": chunk["type"]
#                     }
#                 )
                
#                 split_docs = text_splitter.split_documents([temp_doc])
#                 all_docs_for_embedding.extend(split_docs)

#         if not all_docs_for_embedding:
#             logger.warning("لم يتم العثور على محتوى للتخزين.")
#             return

#         logger.info(f"💾 بدء تخزين {len(all_original_contents)} قطعة محتوى أصلي و {len(all_docs_for_embedding)} قطعة متجهة...")

#         # --- تخزين المحتوى الأصلي ---
#         fs.mset(list(zip(all_doc_ids, all_original_contents)))
#         logger.info("   > ✅ اكتمل تخزين المحتوى الأصلي في مخزن المستندات.")

#         # --- **المنطق الجديد: التخزين بالدفعات الصغيرة والمتحكم بها** ---
#         batch_size = 32  # يمكنك تعديل هذا الرقم، 32 هو بداية جيدة
#         total_batches = (len(all_docs_for_embedding) + batch_size - 1) // batch_size
        
#         logger.info(f"سيتم تقسيم {len(all_docs_for_embedding)} قطعة إلى {total_batches} دفعة (حجم الدفعة: {batch_size}).")

#         with tqdm(total=len(all_docs_for_embedding), desc="تضمين وتخزين المتجهات") as pbar:
#             for i in range(0, len(all_docs_for_embedding), batch_size):
#                 batch_docs = all_docs_for_embedding[i:i + batch_size]
                
#                 # إنشاء معرفات فريدة لهذه الدفعة فقط
#                 batch_ids = [f"{doc.metadata['doc_id']}-{i+j}" for j, doc in enumerate(batch_docs)]
                
#                 try:
#                     vectorstore.add_documents(
#                         documents=batch_docs,
#                         ids=batch_ids
#                     )
#                     pbar.update(len(batch_docs))
#                     time.sleep(0.1) # <-- إضافة فترة راحة بسيطة جدًا
#                 except Exception as batch_err:
#                     logger.error(f"فشل معالجة دفعة تبدأ من العنصر {i}. الخطأ: {batch_err}")
#                     logger.warning("سيتم تخطي هذه الدفعة والمتابعة.")
#                     pbar.update(len(batch_docs)) # تحديث شريط التقدم لتجنب التوقف
#                     continue # الانتقال إلى الدفعة التالية

#         logger.info("   > ✅ اكتمل تخزين المتجهات في قاعدة البيانات.")

#         # --- تحديث سجل الملفات المعالجة ---
#         processed_log.extend(files_to_process)
#         save_processed_files_log(processed_log)
        
#         logger.info("\n🎉 اكتملت عملية التخزين بنجاح!")

#     except Exception as e:
#         logger.error(f"❌ حدث خطأ فادح أثناء عملية التخزين: {e}")

# if __name__ == "__main__":
#     run_ingestion()
    
# project_core/processing/3_run_ingestion.py

import os
import logging
import json
from datetime import datetime
from tqdm import tqdm
import time

# --- استيراد المكتبات الضرورية ---
from langchain_community.vectorstores.chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# --- استيراد الوحدات المخصصة ---
from project_core.core.config import (
    get_embeddings_model,
    VECTORSTORE_PATH,
    DOCSTORE_PATH,
    COLLECTION_NAME,
    BASE_DIR,
    PROCESSED_LOG_FILE,
    ENRICHED_DIR
)
from project_core.processing.utils import load_processed_files_log, save_processed_files_log

# --- إعداد نظام التسجيل ---
LOGS_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
log_filename = datetime.now().strftime(f"ingestion_run_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.FileHandler(os.path.join(LOGS_DIR, log_filename), encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger("ingestion_pipeline")

def run_ingestion():
    logger.info("="*50 + "\n🚀 بدء المرحلة الثالثة: التخزين النهائي (نسخة محسنة للمرشحات)...\n" + "="*50)

    try:
        embeddings_model = get_embeddings_model()
        if not os.path.exists(DOCSTORE_PATH): os.makedirs(DOCSTORE_PATH)
        fs = LocalFileStore(DOCSTORE_PATH)
        
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings_model,
            persist_directory=VECTORSTORE_PATH,
        )
        logger.info("✅ تم تهيئة قواعد البيانات بنجاح.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info("✅ تم تهيئة قاطع النصوص الذكي.")

        # --- **تعديل مهم: حذف سجل الملفات المعالجة لبدء كل شيء من جديد** ---
        if os.path.exists(PROCESSED_LOG_FILE):
            os.remove(PROCESSED_LOG_FILE)
            logger.warning("تم حذف سجل الملفات المعالجة القديم للبدء من جديد.")

        processed_log = load_processed_files_log()
        all_files = [f for f in os.listdir(ENRICHED_DIR) if f.endswith(".json")]
        files_to_process = [f for f in all_files if f not in processed_log]

        if not files_to_process:
            logger.warning("🎉 لا توجد ملفات جديدة للتخزين. كل شيء محدّث!")
            return

        logger.info(f"🔍 تم العثور على {len(files_to_process)} ملفات جديدة تحتاج إلى تخزين.")

        all_docs_for_embedding = []
        all_original_contents = []
        all_doc_ids = []

        for filename in tqdm(files_to_process, desc="قراءة الملفات المثرية"):
            file_path = os.path.join(ENRICHED_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            for i, chunk in enumerate(chunks):
                # إنشاء doc_id فريد ومستقر
                doc_id = f"{chunk['metadata']['tenant_id']}-{os.path.splitext(chunk['metadata']['source_file'])[0]}-{i}"
                
                original_content = chunk["original_content"]
                encoded_content = original_content.encode('utf-8') if isinstance(original_content, str) else original_content
                all_original_contents.append(encoded_content)
                all_doc_ids.append(doc_id)

                enriched_content = chunk["enriched_content"]
                
                # --- **الحل الحاسم هنا** ---
                # نقوم بإنشاء مستند واحد لكل قطعة، مع تمرير البيانات الوصفية الكاملة
                # سيقوم قاطع النصوص بنسخ هذه البيانات الوصفية إلى جميع القطع الناتجة
                metadata = {
                    "doc_id": doc_id,
                    "source_file": chunk["metadata"]["source_file"],
                    "tenant_id": chunk["metadata"]["tenant_id"],
                    "type": chunk["type"]
                }
                
                temp_doc = Document(
                    page_content=enriched_content,
                    metadata=metadata
                )
                
                # الآن نقوم بتقطيع هذا المستند الواحد
                split_docs = text_splitter.split_documents([temp_doc])
                
                # كل قطعة من split_docs ستحتوي الآن على نفس البيانات الوصفية
                all_docs_for_embedding.extend(split_docs)

        if not all_docs_for_embedding:
            logger.warning("لم يتم العثور على محتوى للتخزين.")
            return

        logger.info(f"💾 بدء تخزين {len(all_original_contents)} قطعة محتوى أصلي و {len(all_docs_for_embedding)} قطعة متجهة...")

        fs.mset(list(zip(all_doc_ids, all_original_contents)))
        logger.info("   > ✅ اكتمل تخزين المحتوى الأصلي.")

        batch_size = 32
        total_batches = (len(all_docs_for_embedding) + batch_size - 1) // batch_size
        logger.info(f"سيتم تقسيم {len(all_docs_for_embedding)} قطعة إلى {total_batches} دفعة (حجم الدفعة: {batch_size}).")

        with tqdm(total=len(all_docs_for_embedding), desc="تضمين وتخزين المتجهات") as pbar:
            for i in range(0, len(all_docs_for_embedding), batch_size):
                batch_docs = all_docs_for_embedding[i:i + batch_size]
                
                # معرفات فريدة لكل قطعة متجهة
                batch_ids = [f"{doc.metadata['doc_id']}-{j}" for j, doc in enumerate(batch_docs)]
                
                try:
                    vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                    pbar.update(len(batch_docs))
                    time.sleep(0.1)
                except Exception as batch_err:
                    logger.error(f"فشل معالجة دفعة تبدأ من العنصر {i}. الخطأ: {batch_err}")
                    pbar.update(len(batch_docs))
                    continue

        logger.info("   > ✅ اكتمل تخزين المتجهات في قاعدة البيانات.")

        processed_log.extend(files_to_process)
        save_processed_files_log(processed_log)
        
        logger.info("\n🎉 اكتملت عملية التخزين بنجاح!")

    except Exception as e:
        logger.error(f"❌ حدث خطأ فادح أثناء عملية التخزين: {e}", exc_info=True)

if __name__ == "__main__":
    run_ingestion()
