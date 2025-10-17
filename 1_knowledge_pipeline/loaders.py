# 1_knowledge_pipeline/loaders.py

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)

# قاموس لربط امتداد الملف بالـ Loader المناسب
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
}

def load_documents(source_dir: str) -> List[Document]:
    """
    يقوم بتحميل جميع المستندات المدعومة (PDF, DOCX, TXT) من مجلد محدد.

    Args:
        source_dir (str): المسار إلى المجلد الذي يحتوي على ملفات العميل.

    Returns:
        List[Document]: قائمة من كائنات Document، حيث كل كائن يمثل صفحة أو مستند.
    """
    all_documents = []
    print(f"📂 جارٍ البحث عن المستندات في المسار: '{source_dir}'")

    if not os.path.isdir(source_dir):
        raise ValueError(f"المسار المحدد ليس مجلدًا صالحًا: {source_dir}")

    # المرور على كل الملفات داخل المجلد المحدد
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        # تخطي المجلدات الفرعية والملفات المخفية
        if not os.path.isfile(file_path) or filename.startswith('.'):
            continue

        # تحديد الـ Loader المناسب بناءً على امتداد الملف
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[file_ext]
            print(f"  - 📄 جارٍ تحميل الملف: '{filename}' باستخدام {loader_class.__name__}...")
            
            try:
                # بعض الـ Loaders تتطلب وسائط مختلفة
                if file_ext == ".txt":
                    loader = loader_class(file_path, encoding="utf-8")
                else:
                    loader = loader_class(file_path)
                
                # تحميل محتوى الملف
                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                print(f"    - ✅ تم تحميل {len(loaded_docs)} جزء/صفحة.")

            except Exception as e:
                print(f"    - ❌ فشل تحميل الملف '{filename}'. الخطأ: {e}")
        else:
            print(f"  - ⚠️ تم تخطي ملف غير مدعوم: '{filename}'")

    if not all_documents:
        print("⚠️ لم يتم العثور على أي مستندات مدعومة في المجلد.")
    
    print(f"\n🎉 اكتمل التحميل. إجمالي عدد الأجزاء/الصفحات المحملة: {len(all_documents)}")
    return all_documents

