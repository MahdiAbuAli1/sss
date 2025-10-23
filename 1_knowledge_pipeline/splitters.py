# 1_knowledge_pipeline/splitters.py (النسخة المحدثة)
# -----------------------------------------------------------------------------
# هذه الوحدة مسؤولة عن تقطيع المستندات النظيفة إلى قطع أصغر (chunks).
# -----------------------------------------------------------------------------

from typing import List
from langchain_core.documents import Document
#  --- تم تحديث سطر الاستيراد هذا ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- تعريف إعدادات التقطيع ---
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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "، ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"[*] اكتمل التقطيع. نتج عنه {len(chunks)} قطعة معلومات نهائية.")
    return chunks

