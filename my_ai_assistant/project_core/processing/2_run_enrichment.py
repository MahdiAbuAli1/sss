# project_core/processing/2_run_enrichment.py

import os
import logging
import json
from datetime import datetime
from tqdm import tqdm
import time

# --- استيراد المكتبات الضرورية ---
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- استيراد الوحدات المخصصة من المشروع ---
from project_core.core.config import (
    get_text_enrichment_llm,
    get_multimodal_llm,
    ENABLE_PROCESSING_ENRICHMENT,
    BASE_DIR
)

# --- إعداد نظام التسجيل ---
LOGS_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
log_filename = datetime.now().strftime(f"enrichment_run_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.FileHandler(os.path.join(LOGS_DIR, log_filename), encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger("enrichment_pipeline")

# --- تعريف مجلدات الإدخال والإخراج ---
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate_outputs")
ENRICHED_DIR = os.path.join(BASE_DIR, "enriched_outputs")
if not os.path.exists(ENRICHED_DIR): os.makedirs(ENRICHED_DIR)

# --- تهيئة نماذج وسلاسل المعالجة ---
try:
    enrichment_llm = get_text_enrichment_llm().with_config({"request_timeout": 600}) # 10 دقائق
    multimodal_llm = get_multimodal_llm().with_config({"request_timeout": 600}) # 10 دقائق

    enrichment_prompt = ChatPromptTemplate.from_template(
        """أنت خبير في تنظيم المعلومات. مهمتك هي إعادة هيكلة النص التالي ليكون سهل الفهم ومثالياً لمحرك بحث. استخدم العناوين الواضحة والقوائم النقطية. إذا كان النص يصف خطوات، رقمها. إذا كان يحتوي على تعريفات، أبرزها.

النص الأصلي:
---
{text}
---

النسخة المنظمة والمُثراة باللغة العربية:"""
    )
    enrichment_chain = enrichment_prompt | enrichment_llm | StrOutputParser()

    image_summarize_prompt = ChatPromptTemplate.from_messages(
        [("user", [
            {"type": "text", "text": """أنت خبير تحليل واجهات وتصاميم. صف هذه الصورة كأنك تشرحها لشخص كفيف. كن دقيقًا جدًا.
1.  **ابدأ بالوصف العام:** ما هو نوع هذه الصورة (واجهة تطبيق، مخطط، شعار)؟
2.  **حلل الهيكل:** صف التخطيط (أعلى، وسط، أسفل).
3.  **صف العناصر التفاعلية:** اذكر كل زر، أيقونة، قائمة، أو حقل إدخال. صف شكله، لونه، النص المكتوب عليه، وموقعه الدقيق.
4.  **اقرأ كل النصوص:** اكتب كل نص تراه في الصورة كما هو.
5.  **صف العناصر غير التفاعلية:** اذكر أي صور، شعارات، أو رسوم بيانية أخرى."""},
            {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},
        ])]
    )
    image_summarize_chain = image_summarize_prompt | multimodal_llm | StrOutputParser()

except Exception as e:
    logger.error(f"فشل كارثي في تهيئة نماذج اللغة: {e}")
    enrichment_chain, image_summarize_chain = None, None


def run_enrichment():
    logger.info("="*50 + "\n🚀 بدء مرحلة الإثراء (نسخة قوية مع نقاط حفظ)...\n" + "="*50)

    if not ENABLE_PROCESSING_ENRICHMENT:
        logger.warning("تم تعطيل مرحلة الإثراء. سيتم تخطي هذه المرحلة.")
        return

    if not all([enrichment_chain, image_summarize_chain]):
        logger.error("فشل تهيئة نماذج اللغة. لا يمكن المتابعة.")
        return

    files_to_process = [f for f in os.listdir(INTERMEDIATE_DIR) if f.endswith(".json")]
    
    # --- حلقة رئيسية لمعالجة كل ملف ---
    for filename in tqdm(files_to_process, desc="معالجة الملفات الإجمالية"):
        input_path = os.path.join(INTERMEDIATE_DIR, filename)
        # --- اسم ملف الإخراج النهائي والملف المؤقت (نقطة الحفظ) ---
        final_output_path = os.path.join(ENRICHED_DIR, filename)
        temp_output_path = os.path.join(ENRICHED_DIR, f"temp_{filename}")

        # --- تحقق مما إذا كان الملف النهائي موجودًا بالفعل ---
        if os.path.exists(final_output_path):
            logger.info(f"✅ الملف {filename} تمت معالجته بالفعل. سيتم تخطيه.")
            continue

        logger.info(f"\n--- بدء معالجة الملف: {filename} ---")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- تحميل التقدم المحفوظ إن وجد ---
        enriched_chunks = []
        if os.path.exists(temp_output_path):
            logger.info("🔍 تم العثور على ملف مؤقت. سيتم استئناف العمل...")
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                enriched_chunks = json.load(f)
        
        num_already_processed = len(enriched_chunks)
        logger.info(f"تمت معالجة {num_already_processed} قطعة معلومات من هذا الملف سابقًا.")

        # --- تجميع كل العناصر التي تحتاج إلى معالجة ---
        all_items_to_process = []
        source_file = data.get("source_file")
        tenant_id = data.get("tenant_id")
        
        for item in data.get("extracted_texts", []):
            all_items_to_process.append({"type": "text", "content": item})
        for item in data.get("extracted_tables_html", []):
            all_items_to_process.append({"type": "table", "content": item})
        for item in data.get("extracted_images_base64", []):
            all_items_to_process.append({"type": "image", "content": item})
            
        # --- تخطي العناصر التي تمت معالجتها بالفعل ---
        items_to_process_now = all_items_to_process[num_already_processed:]

        if not items_to_process_now:
            logger.info("لا توجد عناصر جديدة للمعالجة في هذا الملف.")
        else:
            # --- حلقة لمعالجة العناصر المتبقية وحفظ التقدم ---
            with tqdm(total=len(items_to_process_now), desc=f"إثراء {filename[:15]}") as pbar:
                for i, item_data in enumerate(items_to_process_now):
                    item_type = item_data["type"]
                    original_content = item_data["content"]
                    enriched_content = ""
                    
                    try:
                        if item_type in ["text", "table"]:
                            enriched_content = enrichment_chain.invoke({"text": original_content})
                        elif item_type == "image":
                            enriched_content = image_summarize_chain.invoke({"image_base64": original_content})
                        
                        chunk = {
                            "type": item_type,
                            "original_content": original_content,
                            "enriched_content": enriched_content,
                            "metadata": {"source_file": source_file, "tenant_id": tenant_id}
                        }
                        enriched_chunks.append(chunk)

                        # --- نقطة الحفظ: حفظ التقدم في الملف المؤقت بعد كل عنصر ---
                        with open(temp_output_path, 'w', encoding='utf-8') as f:
                            json.dump(enriched_chunks, f, ensure_ascii=False, indent=4)

                    except Exception as e:
                        logger.error(f"❌ فشل معالجة قطعة رقم {num_already_processed + i + 1}: {e}")
                        # في حالة الفشل، نحفظ المحتوى الأصلي لتجنب فقدان البيانات
                        chunk = {
                            "type": item_type,
                            "original_content": original_content,
                            "enriched_content": f"فشل الإثراء: {original_content}",
                            "metadata": {"source_file": source_file, "tenant_id": tenant_id}
                        }
                        enriched_chunks.append(chunk)
                        with open(temp_output_path, 'w', encoding='utf-8') as f:
                            json.dump(enriched_chunks, f, ensure_ascii=False, indent=4)
                    
                    pbar.update(1)

        # --- بعد الانتهاء من الملف بالكامل ---
        # 1. إعادة تسمية الملف المؤقت إلى الاسم النهائي
        os.rename(temp_output_path, final_output_path)
        logger.info(f"✅ اكتمل إثراء الملف {filename}. تم حفظ {len(enriched_chunks)} قطعة معلومات منظمة.")

if __name__ == "__main__":
    run_enrichment()
