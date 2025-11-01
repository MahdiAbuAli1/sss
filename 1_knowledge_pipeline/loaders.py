# 1_knowledge_pipeline/loaders.py (النسخة المعدلة والصحيحة)

import os
import json
from typing import List, Tuple, Optional
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
}

def load_documents(source_dir: str) -> Tuple[List[Document], Optional[str]]:
    """
    يقوم بتحميل جميع المستندات المدعومة مع ضمان قراءة الملفات النصية بترميز UTF-8.
    """
    all_documents = []
    entity_name = None
    config_file_path = os.path.join(source_dir, "config.json")

    print(f"📂 بدء عملية المسح والتحميل في: '{source_dir}'")

    if not os.path.isdir(source_dir):
        raise ValueError(f"المسار المحدد ليس مجلدًا صالحًا: {source_dir}")

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                entity_name = config_data.get("entity_name")
                if entity_name:
                    print(f"  - ✅ تم العثور على هوية العميل: '{entity_name}'")
        except Exception as e:
            print(f"  - ❌ خطأ أثناء قراءة 'config.json': {e}")
    else:
        print(f"  - ⚠️ تحذير: لم يتم العثور على ملف 'config.json'.")

    for filename in os.listdir(source_dir):
        if filename == "config.json" or filename.startswith('.'):
            continue
        
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[file_ext]
            print(f"  - 📄 جاري تحميل '{filename}' باستخدام {loader_class.__name__}...")
            try:
                # --- vvvvvvvvvvvvvvvv هذا هو التعديل المطلوب vvvvvvvvvvvvvvvv ---
                
                # إذا كان الملف نصيًا، استخدم ترميز UTF-8
                if loader_class == TextLoader:
                    loader = loader_class(file_path, encoding='utf-8')
                else:
                    # للملفات الأخرى (PDF, DOCX)، استمر كالمعتاد
                    loader = loader_class(file_path)
                
                # --- ^^^^^^^^^^^^^^^^^^ نهاية التعديل ^^^^^^^^^^^^^^^^^^ ---

                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                print(f"    - ✅ تم تحميل {len(loaded_docs)} صفحة/جزء.")
            except Exception as e:
                print(f"    - ❌ فشل تحميل الملف '{filename}'. الخطأ: {e}")
        else:
            print(f"  - ⏩ تم تخطي ملف غير مدعوم: '{filename}'")

    if not all_documents:
        print("\nلم يتم العثور على أي مستندات قابلة للمعالجة.")
    
    print(f"\n✅ اكتمل التحميل. إجمالي عدد الصفحات/الأجزاء المستخرجة: {len(all_documents)}")
    return all_documents, entity_name
