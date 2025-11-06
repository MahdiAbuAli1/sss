# project_core/processing/pipeline.py

import os
import uuid
import base64
import logging
import io
import zipfile
import json
from typing import List, Tuple

# --- استيراد المكتبات الأساسية ---
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table
from pdf2image import convert_from_path
from tqdm import tqdm

# --- استيراد الوحدات المخصصة من المشروع ---
from project_core.core.config import (
    get_text_enrichment_llm,
    get_multimodal_llm,
    ENABLE_PROCESSING_ENRICHMENT,
    BASE_DIR,
)

logger = logging.getLogger("pipeline")

# --- تهيئة النماذج وسلاسل المعالجة (Chains) مع "الوضع الآمن" ---
try:
    # إضافة مهلة زمنية (timeout) للنماذج لمنع التجمد
    enrichment_llm = get_text_enrichment_llm().with_config({"request_timeout": 300}) # 5 دقائق
    multimodal_llm = get_multimodal_llm().with_config({"request_timeout": 300}) # 5 دقائق

    # القالب "الذكي" لإثراء النصوص
    enrichment_prompt_template = ChatPromptTemplate.from_template(
        """أنت محلل معلومات خبير. مهمتك هي تحويل النص التالي إلى ملخص منظم ومُثرى بالمعلومات. ركز على استخراج وتوضيح: 1. الخطوات والإجراءات. 2. الأوامر أو الأكواد البرمجية. 3. التعريفات والمفاهيم الأساسية. 4. الأرقام والبيانات الهامة. هدفنا هو إنشاء نسخة من النص تكون مثالية لمحرك بحث دلالي.\n\nالنص الأصلي:\n---\n{text}\n---\n\nالملخص المُنظم والمُثرى باللغة العربية:"""
    )
    enrichment_chain = enrichment_prompt_template | enrichment_llm | StrOutputParser()

    # القالب "الدقيق" لوصف الصور
    image_summarize_prompt_template = ChatPromptTemplate.from_messages(
        [("user", [{"type": "text", "text": "أنت خبير في تحليل واجهات المستخدم. صف هذه الصورة بدقة متناهية. ركز على: 1. النصوص الظاهرة. 2. وصف الأيقونات وألوانها ومواقعها. 3. وصف الأزرار وما هو مكتوب عليها. 4. أي رسوم بيانية أو بيانات معروضة."}, {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},])]
    )
    image_summarize_chain = image_summarize_prompt_template | multimodal_llm | StrOutputParser()

except Exception as e:
    logger.error(f"فشل كارثي في تهيئة نماذج اللغة: {e}")
    enrichment_chain, image_summarize_chain = None, None

# --- الدوال المساعدة ---

def image_summarize(img_base64: str) -> str:
    """ يستدعي سلسلة وصف الصور مع معالجة الأخطاء. """
    if not image_summarize_chain: return "فشل تهيئة سلسلة وصف الصور."
    try:
        return image_summarize_chain.invoke({"image_base64": img_base64})
    except Exception as e:
        logger.error(f"فشل وصف صورة. الخطأ: {e}")
        return "فشل وصف الصورة بسبب انتهاء المهلة أو خطأ آخر."

def extract_images_from_docx(file_path: str) -> List[str]:
    """ الخطة "ج": تستخلص الصور مباشرة من ملفات .docx. """
    images_base64 = []
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            image_files = [f for f in zf.namelist() if f.startswith('word/media/')]
            for filename in tqdm(image_files, desc="استخلاص الصور من DOCX"):
                with zf.open(filename) as f:
                    image_bytes = f.read()
                    images_base64.append(base64.b64encode(image_bytes).decode('utf-8'))
        if images_base64:
            logger.info(f"الخطة 'ج' نجحت: تم استخلاص {len(images_base64)} صورة مباشرة من ملف DOCX.")
    except Exception as e:
        logger.error(f"فشلت الخطة 'ج' لاستخلاص الصور من DOCX. الخطأ: {e}")
    return images_base64

# --- الدالة الرئيسية للمعالجة ---

def process_document_elements(file_path: str, tenant_id: str) -> Tuple[List[Document], List[str], List[bytes]]:
    logger.info(f"بدء معالجة الملف: {os.path.basename(file_path)} للنظام: [{tenant_id}]")
    texts, tables_html, images_base64 = [], [], []

    # المرحلة 1: الاستخلاص الأساسي للنصوص والجداول
    try:
        raw_elements = partition(file_path, strategy="auto", languages=["ara", "eng"])
        for element in raw_elements:
            if isinstance(element, Table) and hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                tables_html.append(element.metadata.text_as_html)
            elif len(str(element).strip()) > 20:
                texts.append(str(element))
    except Exception as e:
        logger.error(f"فشل الاستخلاص الأساسي للملف {file_path}. الخطأ: {e}")

    # المرحلة 2: تفعيل الخطط البديلة القوية لاستخلاص الصور
    if file_path.lower().endswith(".pdf"):
        try:
            pil_images = convert_from_path(file_path)
            if pil_images:
                logger.info(f"الخطة 'ب' (PDF): تحويل {len(pil_images)} صفحة إلى صور.")
                for img in pil_images:
                    buffer = io.BytesIO(); img.save(buffer, format="JPEG");
                    images_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        except Exception as e:
            logger.error(f"فشلت الخطة 'ب' لتحويل PDF إلى صور. الخطأ: {e}")
    elif file_path.lower().endswith(".docx"):
        images_base64.extend(extract_images_from_docx(file_path))

    # --- حفظ مخرجات المراجعة الأولية ---
    DEBUG_DIR = os.path.join(BASE_DIR, "debug_outputs")
    if not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR)
    debug_filename_base = f"{tenant_id}_{os.path.splitext(os.path.basename(file_path))[0]}"
    with open(os.path.join(DEBUG_DIR, f"{debug_filename_base}_initial.json"), 'w', encoding='utf-8') as f:
        json.dump({"stage": "initial_extraction", "texts_count": len(texts), "tables_count": len(tables_html), "images_count": len(images_base64)}, f, ensure_ascii=False, indent=4)

    # المرحلة 3: الإثراء والتنظيم (مع مؤشر تقدم ومعالجة بالدفعات)
    text_enriched, table_enriched, image_summaries = [], [], []
    if ENABLE_PROCESSING_ENRICHMENT:
        logger.info("[الإثراء] بدء إثراء النصوص والجداول ووصف الصور (قد يستغرق هذا وقتاً طويلاً)...")
        try:
            if texts and enrichment_chain:
                text_enriched = enrichment_chain.batch(tqdm(texts, desc="إثراء النصوص"), {"max_concurrency": 5})
            if tables_html and enrichment_chain:
                table_enriched = enrichment_chain.batch(tqdm(tables_html, desc="إثراء الجداول"), {"max_concurrency": 5})
            if images_base64 and image_summarize_chain:
                image_summaries = [image_summarize(img) for img in tqdm(images_base64, desc="وصف الصور")]
        except Exception as e:
            logger.error(f"حدث خطأ أثناء مرحلة الإثراء: {e}. سيتم استخدام المحتوى الأصلي.")
            text_enriched, table_enriched, image_summaries = texts, tables_html, ["فشل وصف الصورة" for _ in images_base64]
    else:
        text_enriched, table_enriched, image_summaries = texts, tables_html, ["صورة من المستند" for _ in images_base64]

    # --- حفظ مخرجات المراجعة بعد الإثراء ---
    with open(os.path.join(DEBUG_DIR, f"{debug_filename_base}_enriched.json"), 'w', encoding='utf-8') as f:
        json.dump({"stage": "enrichment", "enriched_texts": text_enriched, "enriched_tables": table_enriched, "image_summaries": image_summaries}, f, ensure_ascii=False, indent=4)

    # المرحلة 4: التجميع النهائي للتخزين
    documents_for_embedding, doc_ids, original_contents_bytes = [], [], []
    all_enriched = text_enriched + table_enriched + image_summaries
    all_original = texts + tables_html + images_base64
    
    for i, content in enumerate(all_enriched):
        doc_id = str(uuid.uuid4())
        documents_for_embedding.append(Document(page_content=content, metadata={"doc_id": doc_id, "tenant_id": tenant_id, "source_file": os.path.basename(file_path)}))
        original_content = all_original[i]
        # تحويل كل المحتوى الأصلي إلى bytes للتخزين الموحد
        if isinstance(original_content, str):
            original_contents_bytes.append(original_content.encode('utf-8'))
        else: # يفترض أن يكون bytes بالفعل (من الصور)
             original_contents_bytes.append(base64.b64decode(original_content))
        doc_ids.append(doc_id)

    logger.info(f"اكتملت المعالجة بنجاح. تم تجهيز {len(documents_for_embedding)} عنصر للتخزين.")
    return documents_for_embedding, doc_ids, original_contents_bytes
