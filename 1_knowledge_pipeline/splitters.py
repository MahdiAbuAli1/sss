# # 1_knowledge_pipeline/splitters.py (النسخة المحدثة)
# # -----------------------------------------------------------------------------
# # هذه الوحدة مسؤولة عن تقطيع المستندات النظيفة إلى قطع أصغر (chunks).
# # -----------------------------------------------------------------------------

# from typing import List
# from langchain_core.documents import Document
# #  --- تم تحديث سطر الاستيراد هذا ---
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # --- تعريف إعدادات التقطيع ---
# CHUNK_SIZE = 1500 
# CHUNK_OVERLAP = 250

# def split_documents(documents: List[Document]) -> List[Document]:
#     """
#     تقوم بتقسيم قائمة من المستندات إلى قطع أصغر باستخدام RecursiveCharacterTextSplitter.

#     Args:
#         documents (List[Document]): قائمة المستندات النظيفة.

#     Returns:
#         List[Document]: قائمة جديدة من القطع (Chunks).
#     """
#     print(f"\n[+] المرحلة 3: تقطيع {len(documents)} جزء/صفحة إلى قطع أصغر...")
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", "، ", " ", ""]
#     )

#     chunks = text_splitter.split_documents(documents)

#     print(f"[*] اكتمل التقطيع. نتج عنه {len(chunks)} قطعة معلومات نهائية.")
#     return chunks

# 1_knowledge_pipeline/splitters.py (النسخة النهائية - v2.0 مع دمج الصفحات)

from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. تعريف الإعدادات النهائية للتقطيع ---
# هذا هو القرار الهندسي الذي توصلنا إليه من الاختبارات
CHUNK_SIZE = 1300
CHUNK_OVERLAP = 250

# --- 2. المُقطِّع النهائي ---
# يتم تعريفه مرة واحدة لاستخدامه في كل مكان.
final_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "، ", " ", ""],
    keep_separator=True
)

# --- 3. الدالة الرئيسية للتقطيع (التي تطبق دمج الصفحات) ---
def split_documents(initial_pages: List[Document]) -> List[Document]:
    """
    تعالج قائمة من الصفحات الأولية، تقوم بدمج المحتوى لكل ملف مصدر،
    ثم تقطيعه بشكل ذكي للحفاظ على السياق عبر الصفحات.

    Args:
        initial_pages (List[Document]): قائمة الصفحات الأولية المحملة من loaders.py.

    Returns:
        List[Document]: قائمة نهائية من المقاطع (Chunks) الجاهزة للتضمين.
    """
    print(f"\n[+] المرحلة 2: بدء دمج وتقطيع {len(initial_pages)} صفحة أولية...")

    if not initial_pages:
        print("[*] لا توجد صفحات للتقطيع. تم التخطي.")
        return []

    # الخطوة 1: تجميع محتوى الصفحات حسب الملف المصدر
    # هذا هو جوهر حل "الفجوة السياقية بين الصفحات"
    content_by_source: Dict[str, str] = {}
    metadata_by_source: Dict[str, Dict] = {}

    for page in initial_pages:
        source = page.metadata.get("source", "unknown_source")
        if source not in content_by_source:
            content_by_source[source] = ""
            # نحتفظ بالميتا-بيانات الأصلية للملف (مثل tenant_id, entity_name)
            metadata_by_source[source] = page.metadata
        
        # دمج محتوى الصفحات مع فاصل واضح للحفاظ على بنية النص
        content_by_source[source] += page.page_content + "\n\n"

    # الخطوة 2: إنشاء كائنات Document مدمجة لكل ملف مصدر
    merged_documents = []
    for source, content in content_by_source.items():
        merged_doc = Document(
            page_content=content,
            metadata=metadata_by_source[source] # استخدام الميتا-بيانات الأصلية
        )
        merged_documents.append(merged_doc)
        
    print(f"[*] تم دمج الصفحات في {len(merged_documents)} مستند منطقي (ملف).")

    # الخطوة 3: تطبيق المُقطِّع النهائي على المستندات المدمجة
    print(f"[*] جاري تطبيق التقطيع (حجم: {CHUNK_SIZE}, تداخل: {CHUNK_OVERLAP})...")
    final_chunks = final_text_splitter.split_documents(merged_documents)

    print(f"[*] اكتمل التقطيع. نتج عنه {len(final_chunks)} قطعة معلومات نهائية.")
    
    return final_chunks
