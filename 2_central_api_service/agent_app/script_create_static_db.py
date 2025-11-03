# ==============================================================================
# اسم الملف: script_create_static_db.py
# الوصف: يقوم هذا السكربت بقراءة ملفات JSON التي تحتوي على أسئلة وأجوبة
#        ثابتة، ويقوم بإنشاء قاعدة بيانات متجهة (Vector Database) باستخدام FAISS
#        لفهرستها دلالياً. يتم حفظ قاعدة البيانات محلياً لاستخدامها
#        في نظام التوجيه الدلالي (Semantic Router).
# ==============================================================================

import os
import json
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# --- 1. إعدادات التسجيل (Logging) ---
# لإظهار رسائل واضحة حول ما يحدث أثناء تشغيل السكربت
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. الإعدادات الأساسية والمسارات الديناميكية ---

# هذا السطر السحري يحدد تلقائياً مسار المجلد الذي يوجد فيه هذا السكربت
# مثال: C:\Users\mahdi\support_service_platform\2_central_api_service\agent_app
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # هذا الحل البديل يعمل في بيئات مثل Jupyter notebooks
    SCRIPT_DIR = os.getcwd()

# المسار إلى مجلد البيانات هو "static_responses" الموجود بجانب السكربت
DATA_DIR = os.path.join(SCRIPT_DIR, "static_responses")

# المسار الذي سيتم حفظ قاعدة البيانات فيه (سيتم إنشاء مجلد "static_db" بجانب السكربت)
DB_SAVE_PATH = os.path.join(SCRIPT_DIR, "static_db")

# إعدادات نموذج التضمين و Ollama
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434" )


# --- 3. تحديد فئات البيانات ---
# هذا القاموس يربط كل ملف JSON بالفئة الدلالية التي ينتمي إليها
FILE_TO_CATEGORY = {
    "compliments_conversations_dataset.json": "compliment",
    "confirmation_queries.json": "confirmation",
    "dataset_inappropriate_responses.json": "inappropriate",
    "dataset_interactive_responses.json": "small_talk",
    "farewell_prayers_dataset.json": "farewell",
    "insults_and_responses.json": "insult",
    "random_words_dataset.json": "random",
    "thanks_phrases.json": "thanks",
    "ai_test_questions.json": "ai_test" # إضافة الملف المتبقي
}

# --- 4. الدالة الرئيسية لتنفيذ العملية ---
def main():
    """
    الدالة الرئيسية التي تقوم بتنفيذ جميع خطوات إنشاء قاعدة البيانات.
    """
    logging.info("--- بدء عملية إنشاء قاعدة بيانات الردود الثابتة ---")
    
    all_documents = []
    logging.info(f"البحث عن ملفات البيانات في المجلد: {DATA_DIR}")

    # --- 4.1. قراءة ومعالجة ملفات JSON ---
    for filename, category in FILE_TO_CATEGORY.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logging.warning(f"الملف '{filepath}' غير موجود، سيتم تخطيه.")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # عداد للمستندات في كل ملف
            doc_count_in_file = 0
            for item in data:
                question = item.get("question")
                answers = item.get("answers")
                
                # التأكد من وجود السؤال والإجابات لتجنب الأخطاء
                if question and isinstance(question, str) and answers and isinstance(answers, list):
                    doc = Document(
                        page_content=question,
                        metadata={"answers": answers, "category": category}
                    )
                    all_documents.append(doc)
                    doc_count_in_file += 1
            
            logging.info(f"تم تحميل {doc_count_in_file} مستند من الملف '{filename}'")

        except json.JSONDecodeError:
            logging.error(f"خطأ في تنسيق JSON في الملف '{filepath}'. يرجى التأكد من صحته.")
        except Exception as e:
            logging.error(f"حدث خطأ غير متوقع أثناء معالجة الملف '{filepath}': {e}")

    # --- 4.2. التحقق من وجود مستندات قبل المتابعة ---
    if not all_documents:
        logging.error("فشل تحميل المستندات. لم يتم العثور على أي بيانات صالحة للمعالجة.")
        logging.error("يرجى التأكد من أن مجلد 'static_responses' يحتوي على ملفات JSON بالأسماء الصحيحة وأنها ليست فارغة.")
        return # إنهاء السكربت إذا لم يتم العثور على بيانات

    logging.info(f"إجمالي عدد المستندات التي سيتم فهرستها: {len(all_documents)}")

    # --- 4.3. إنشاء الفهرس الدلالي (Vector Index) ---
    try:
        logging.info(f"بدء تهيئة نموذج التضمين '{EMBEDDING_MODEL}' من خلال Ollama...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        
        logging.info("بدء عملية الفهرسة باستخدام FAISS. قد تستغرق هذه العملية بعض الوقت...")
        static_vector_store = FAISS.from_documents(all_documents, embeddings)
        
        logging.info(f"اكتملت الفهرسة. جاري حفظ قاعدة البيانات في المسار: {DB_SAVE_PATH}")
        static_vector_store.save_local(DB_SAVE_PATH)
        
        logging.info("=" * 50)
        logging.info("✅ نجحت العملية! ✅")
        logging.info(f"تم إنشاء وحفظ قاعدة بيانات الردود الثابتة بنجاح في '{DB_SAVE_PATH}'.")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"حدث خطأ فادح أثناء عملية الفهرسة أو الحفظ: {e}")
        logging.error("تأكد من أن خدمة Ollama تعمل وأن النموذج المطلوب متاح.")

# --- 5. نقطة انطلاق السكربت ---
if __name__ == "__main__":
    main()
