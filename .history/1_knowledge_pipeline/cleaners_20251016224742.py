# 1_knowledge_pipeline/cleaners.py
# -----------------------------------------------------------------------------
# هذه الوحدة مسؤولة عن تنظيف المحتوى النصي للمستندات المحملة.
# يزيل الشوائب الشائعة مثل المسافات الزائدة وفواصل الأسطر غير المرغوب فيها.
# -----------------------------------------------------------------------------

import re
from typing import List
from langchain_core.documents import Document

def clean_documents(documents: List[Document]) -> List[Document]:
    """
    يقوم بتنفيذ سلسلة من عمليات التنظيف على محتوى كل مستند في القائمة.

    Args:
        documents (List[Document]): قائمة المستندات المحملة.

    Returns:
        List[Document]: القائمة نفسها بعد تحديث محتوى المستندات.
    """
    print(f"\n[+] المرحلة 2: تنظيف {len(documents)} جزء/صفحة...")
    
    for doc in documents:
        content = doc.page_content
        
        # 1. إزالة المسافات البيضاء الزائدة من البداية والنهاية
        content = content.strip()
        
        # 2. استبدال 3 فواصل أسطر أو أكثر بفاصلين فقط للحفاظ على الفقرات
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 3. استبدال المسافات المتعددة بمسافة واحدة
        content = re.sub(r' +', ' ', content)
        
        # 4. دمج الأسطر المكسورة (شائع في PDF) مع الحفاظ على نهايات الجمل والفقرات
        # هذا التعبير النمطي يزيل فاصل السطر إذا لم يكن مسبوقًا بنقطة.
        content = re.sub(r'(?<!\.)\n(?!\n)', ' ', content)

        # تحديث محتوى المستند
        doc.page_content = content
            
    print("[*] اكتمل تنظيف النصوص.")
    return documents
