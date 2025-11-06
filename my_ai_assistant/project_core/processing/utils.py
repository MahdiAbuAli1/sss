# project_core/processing/utils.py

import os
import json
import logging

# استيراد المجلد الأساسي من الإعدادات
from project_core.core.config import BASE_DIR

# إعداد مسجل بسيط لهذا الملف
logger = logging.getLogger("processing_utils")

# تعريف مسار ملف السجل
PROCESSED_LOG_FILE = os.path.join(BASE_DIR, "storage", "processed_files.log")

def load_processed_files_log():
    """
    تحميل قائمة بأسماء الملفات التي تمت معالجتها وتخزينها سابقًا.
    إذا لم يكن الملف موجودًا، يتم إرجاع قائمة فارغة.
    """
    if os.path.exists(PROCESSED_LOG_FILE):
        try:
            with open(PROCESSED_LOG_FILE, 'r', encoding='utf-8') as f:
                # قراءة كل سطر وإزالة أي مسافات بيضاء أو أسطر فارغة
                processed_files = [line.strip() for line in f if line.strip()]
            logger.info(f"تم تحميل سجل المعالجة. {len(processed_files)} ملفات تمت معالجتها سابقًا.")
            return processed_files
        except Exception as e:
            logger.error(f"فشل تحميل سجل المعالجة: {e}. سيتم البدء من جديد.")
            return []
    return []

def save_processed_files_log(processed_files):
    """
    حفظ القائمة المحدثة لأسماء الملفات المعالجة في ملف السجل.
    """
    try:
        # التأكد من أن مجلد storage موجود
        os.makedirs(os.path.dirname(PROCESSED_LOG_FILE), exist_ok=True)
        with open(PROCESSED_LOG_FILE, 'w', encoding='utf-8') as f:
            for filename in processed_files:
                f.write(filename + '\n')
        logger.info(f"تم تحديث سجل المعالجة بنجاح. إجمالي الملفات المعالجة: {len(processed_files)}")
    except Exception as e:
        logger.error(f"فشل حفظ سجل المعالجة: {e}")

