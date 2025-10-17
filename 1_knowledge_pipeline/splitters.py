# 1_knowledge_pipeline/splitters.py
# -----------------------------------------------------------------------------
# هذه الوحدة مسؤولة عن تقطيع المستندات النظيفة إلى قطع أصغر (chunks).
# استخدام مقسم هرمي يضمن أن القطع تحافظ على السياق قدر الإمكان
# ضمن حدود الحجم المحدد.
# -----------------------------------------------------------------------------

from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- تعريف إعدادات التقطيع ---
# يمكن تعديل هذه القيم لاحقًا بناءً على نتائج التجارب.
# chunk_size: الحجم الأقصى لكل قطعة (بالأحرف).
# chunk_overlap: عدد الأحرف المتداخلة بين القطع المتتالية للحفاظ على السياق.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

def split_documents(documents: List[Document]) -> List[Document]:
    """
    تقوم بتقسيم قائمة من المستندات إلى قطع أصغر باستخدام RecursiveCharacterTextSplitter.

    Args:
        documents (List[Document]): قائمة المستندات النظيفة.

    Returns:
        List[Document]: قائمة جديدة من القطع (Chunks).
    """
    print(f"\n[+] المرحلة 3: تقطيع {len(documents)} جزء/صفحة إلى قطع أصغر...")
    
    # إعداد المـُقسـِّم (Splitter)
    # يستخدم قائمة من الفواصل بالترتيب لمحاولة الحفاظ على تماسك النص.
    # يبدأ بالفقرات، ثم الأسطر، ثم الجمل، وهكذا.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "، ", " ", ""]
    )

    # استخدام دالة split_documents التي تتعامل مع قائمة من المستندات مباشرة
    chunks = text_splitter.split_documents(documents)

    print(f"[*] اكتمل التقطيع. نتج عنه {len(chunks)} قطعة معلومات نهائية.")
    return chunks
