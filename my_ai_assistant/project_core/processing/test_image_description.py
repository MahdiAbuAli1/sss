# project_core/processing/test_image_description.py

import os
import json
import logging
import random

# --- استيراد الوحدات الضرورية ---
from project_core.core.config import BASE_DIR
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from project_core.core.config import get_multimodal_llm
from tqdm import tqdm

# --- إعداد أساسي للسجلات لرؤية أي أخطاء ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("image_test")

# --- تعريف المسارات ---
TEST_JSON_FILE = "perfume_shop_01_project.json" 
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate_outputs")
JSON_FILE_PATH = os.path.join(INTERMEDIATE_DIR, TEST_JSON_FILE)

# --- عدد الصور المراد اختبارها ---
NUM_IMAGES_TO_TEST = 5

def test_multiple_image_descriptions():
    logger.info("="*50)
    logger.info(f"🚀 بدء اختبار وصف {NUM_IMAGES_TO_TEST} صور مختلفة...")
    logger.info("="*50)

    # 1. تحميل النموذج وسلسلة المعالجة
    try:
        multimodal_llm = get_multimodal_llm().with_config({"request_timeout": 300})
        image_summarize_prompt = ChatPromptTemplate.from_messages(
            [("user", [{"type": "text", "text": "صف هذه الصورة بدقة متناهية. ركز على كل التفاصيل المرئية، بما في ذلك النصوص، الأيقونات، والألوان."}, {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},])]
        )
        image_summarize_chain = image_summarize_prompt | multimodal_llm | StrOutputParser()
        logger.info("✅ تم تهيئة نموذج وصف الصور بنجاح.")
    except Exception as e:
        logger.error(f"❌ فشل في تهيئة نموذج اللغة: {e}")
        return

    # 2. قراءة ملف JSON واختيار عينة من الصور
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_images_b64 = data.get("extracted_images_base64", [])
        if not all_images_b64:
            logger.warning(f"لم يتم العثور على صور في الملف: {TEST_JSON_FILE}")
            return
        
        total_images = len(all_images_b64)
        logger.info(f"🔬 تم العثور على {total_images} صورة. سيتم اختيار {NUM_IMAGES_TO_TEST} صور للاختبار.")

        # اختيار فهارس صور متباعدة بشكل استراتيجي
        if total_images <= NUM_IMAGES_TO_TEST:
            # إذا كان عدد الصور أقل من أو يساوي المطلوب، نختبرها كلها
            indices_to_test = list(range(total_images))
        else:
            # اختيار عينة عشوائية من الفهارس
            indices_to_test = sorted(random.sample(range(total_images), NUM_IMAGES_TO_TEST))
        
        images_to_test = [(i, all_images_b64[i]) for i in indices_to_test]
        
    except Exception as e:
        logger.error(f"❌ فشل في قراءة ملف JSON أو اختيار الصور: {e}")
        return

    # 3. استدعاء النموذج لكل صورة وطباعة النتائج
    logger.info("🧠 يتم الآن إرسال الصور إلى النموذج للوصف...")
    
    for index, image_b64 in images_to_test:
        try:
            print("\n" + "="*20 + f" وصف الصورة رقم {index} " + "="*20)
            
            # استخدام tqdm لعرض مؤشر انتظار بسيط لكل صورة
            with tqdm(total=1, desc=f"جاري وصف الصورة {index}") as pbar:
                description = image_summarize_chain.invoke({"image_base64": image_b64})
                pbar.update(1)
            
            print(description)
            
        except Exception as e:
            logger.error(f"❌ فشل وصف الصورة رقم {index}. الخطأ: {e}")
            print(f"فشل وصف الصورة رقم {index}.")

    logger.info("\n" + "="*50)
    logger.info("🎉 اكتمل اختبار وصف الصور بنجاح!")
    logger.info("="*50)


if __name__ == "__main__":
    test_multiple_image_descriptions()
