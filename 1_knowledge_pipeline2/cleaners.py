# 1_knowledge_pipeline/cleaners.py (نسخة محدثة وقوية)
import re
from typing import List
from langchain_core.documents import Document
import arabic_reshaper
from bidi.algorithm import get_display

def clean_documents(documents: List[Document]) -> List[Document]:
    """
    يقوم بتنفيذ سلسلة من عمليات التنظيف على محتوى كل مستند في القائمة.
    """
    print(f"\n[+] المرحلة 2: تنظيف {len(documents)} جزء/صفحة...")
    
    for doc in documents:
        content = doc.page_content
        
        # --- الإصلاح الجذري لمشكلة PDF العربية ---
        # 1. إعادة تشكيل الحروف العربية المتفرقة
        reshaped_text = arabic_reshaper.reshape(content)
        # 2. تصحيح اتجاه النص من اليمين إلى اليسار
        bidi_text = get_display(reshaped_text)
        content = bidi_text
        
        # --- عمليات التنظيف القياسية ---
        # 3. إزالة المسافات البيضاء الزائدة من البداية والنهاية
        content = content.strip()
        
        # 4. استبدال 3 فواصل أسطر أو أكثر بفاصلين فقط للحفاظ على الفقرات
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 5. استبدال المسافات المتعددة بمسافة واحدة
        content = re.sub(r' +', ' ', content)

        # 6. إزالة المسافات بين الكلمات العربية والحروف اللاتينية المتصلة
        content = re.sub(r'([_,-])', ' ', content)
        content = re.sub(r'([ا-ي])([A-Za-z])', r'\1 \2', content)
        content = re.sub(r'([A-Za-z])([ا-ي])', r'\1 \2', content)

        # تحديث محتوى المستند
        doc.page_content = content
            
    print("[*] اكتمل تنظيف النصوص.")
    return documents
