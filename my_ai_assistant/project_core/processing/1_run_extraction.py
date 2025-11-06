# project_core/processing/1_run_extraction.py
#في النهاية، ستجد مجلدًا جديدًا اسمه intermediate_outputs، وبداخله ملفات .json تحتوي على كل النصوص والجداول والصور المستخلصة.
import os
import logging
import json
import base64
import io
import zipfile
from datetime import datetime

# استيراد المكتبات الضرورية
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table
from pdf2image import convert_from_path
from tqdm import tqdm

from project_core.core.config import DATA_SOURCES_DIR, BASE_DIR

# --- إعداد نظام التسجيل ---
LOGS_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
log_filename = datetime.now().strftime(f"extraction_run_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.FileHandler(os.path.join(LOGS_DIR, log_filename), encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger("extraction_pipeline")

# --- مجلد المخرجات الوسيطة ---
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate_outputs")
if not os.path.exists(INTERMEDIATE_DIR): os.makedirs(INTERMEDIATE_DIR)

def extract_images_from_docx(file_path: str):
    # ... (نفس الدالة من الكود السابق) ...
    images_base64 = []
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            image_files = [f for f in zf.namelist() if f.startswith('word/media/')]
            for filename in tqdm(image_files, desc="استخلاص الصور من DOCX"):
                with zf.open(filename) as f:
                    images_base64.append(base64.b64encode(f.read()).decode('utf-8'))
        if images_base64:
            logger.info(f"الخطة 'ج' نجحت: تم استخلاص {len(images_base64)} صورة من DOCX.")
    except Exception as e: logger.error(f"فشلت الخطة 'ج' لاستخلاص الصور من DOCX. الخطأ: {e}")
    return images_base64

def run_extraction():
    logger.info("="*50 + "\n🚀 بدء مرحلة الاستخلاص فقط...\n" + "="*50)
    
    for tenant_id in os.listdir(DATA_SOURCES_DIR):
        tenant_path = os.path.join(DATA_SOURCES_DIR, tenant_id)
        if not os.path.isdir(tenant_path): continue
        
        logger.info(f"\n--- المسح داخل مجلد النظام: [{tenant_id}] ---")
        for file_name in os.listdir(tenant_path):
            file_path = os.path.join(tenant_path, file_name)
            if not os.path.isfile(file_path): continue

            logger.info(f"--- بدء استخلاص الملف: {file_name} ---")
            texts, tables_html, images_base64 = [], [], []

            # 1. استخلاص النصوص والجداول
            try:
                raw_elements = partition(file_path, strategy="auto", languages=["ara", "eng"])
                for element in raw_elements:
                    if isinstance(element, Table) and hasattr(element.metadata, 'text_as_html'):
                        tables_html.append(element.metadata.text_as_html)
                    elif len(str(element).strip()) > 20:
                        texts.append(str(element))
            except Exception as e:
                logger.error(f"فشل استخلاص النصوص/الجداول من {file_name}. الخطأ: {e}")

            # 2. استخلاص الصور بالخطط البديلة
            if file_path.lower().endswith(".pdf"):
                try:
                    pil_images = convert_from_path(file_path)
                    for img in tqdm(pil_images, desc="تحويل صفحات PDF إلى صور"):
                        buffer = io.BytesIO(); img.save(buffer, format="JPEG")
                        images_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
                except Exception as e:
                    logger.error(f"فشل تحويل PDF إلى صور لـ {file_name}. الخطأ: {e}")
            elif file_path.lower().endswith(".docx"):
                images_base64.extend(extract_images_from_docx(file_path))

            # 3. حفظ النتائج في ملف وسيط
            output_data = {
                "source_file": file_name,
                "tenant_id": tenant_id,
                "extracted_texts": texts,
                "extracted_tables_html": tables_html,
                "extracted_images_base64": images_base64
            }
            
            output_filename = f"{tenant_id}_{os.path.splitext(file_name)[0]}.json"
            output_path = os.path.join(INTERMEDIATE_DIR, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
                
            logger.info(f"✅ اكتمل استخلاص الملف {file_name}. تم حفظ النتائج في: {output_path}")
            logger.info(f"   > ملخص: {len(texts)} نص، {len(tables_html)} جدول، {len(images_base64)} صورة.")

if __name__ == "__main__":
    run_extraction()
