# verify_db.py

import os
import json
import logging

# --- استيراد المكتبات الضرورية ---
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore

# --- استيراد الوحدات المخصصة ---
# سنستورد فقط المتغيرات الموجودة بالفعل في config.py
from project_core.core.config import (
    VECTORSTORE_PATH,
    DOCSTORE_PATH,
    COLLECTION_NAME,
    BASE_DIR  # الأهم هو المجلد الأساسي
)

# --- إعداد نظام التسجيل ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- **الحل هنا: تعريف المجلدات محليًا** ---
ENRICHED_DIR = os.path.join(BASE_DIR, "enriched_outputs")

def verify_databases():
    """
    يتحقق من عدد العناصر في قواعد البيانات ويقارنها بالملفات المصدر.
    """
    print("\n--- بدء عملية التحقق من اكتمال التخزين ---")

    # 1. حساب العدد المتوقع من الملفات المصدر
    total_expected_chunks = 0
    try:
        if not os.path.exists(ENRICHED_DIR):
             print(f"❌ خطأ: المجلد '{ENRICHED_DIR}' غير موجود. تأكد من اكتمال مرحلة الإثراء.")
             return

        source_files = [f for f in os.listdir(ENRICHED_DIR) if f.endswith(".json")]
        if not source_files:
            print("⚠️ لم يتم العثور على ملفات مُثراة في 'enriched_outputs'. لا يمكن التحقق.")
            return

        for filename in source_files:
            with open(os.path.join(ENRICHED_DIR, filename), 'r', encoding='utf-8') as f:
                total_expected_chunks += len(json.load(f))
        print(f"🔍 العدد الإجمالي المتوقع للقطع الأصلية (من enriched_outputs): {total_expected_chunks}")
    except Exception as e:
        print(f"❌ فشل في قراءة الملفات المصدر: {e}")
        return

    # 2. التحقق من مخزن المستندات الأصلي (Doc Store)
    try:
        if not os.path.exists(DOCSTORE_PATH):
            print(f"❌ خطأ: مجلد مخزن المستندات '{DOCSTORE_PATH}' غير موجود.")
        else:
            # fs.yield_keys() قد لا تكون الطريقة الأكثر موثوقية للعد
            # الطريقة الأبسط هي عد الملفات مباشرة
            stored_files = os.listdir(DOCSTORE_PATH)
            doc_store_count = len(stored_files)
            print(f"✅ عدد العناصر في مخزن المستندات (Doc Store): {doc_store_count}")
            if doc_store_count == total_expected_chunks:
                print("   > 👍 ممتاز! العدد مطابق للعدد المتوقع.")
            else:
                print(f"   > ⚠️ غير مطابق! (المتوقع: {total_expected_chunks}). قد تكون العملية لم تكتمل.")
    except Exception as e:
        print(f"❌ فشل في الوصول إلى مخزن المستندات: {e}")

    # 3. التحقق من قاعدة البيانات المتجهة (Vector Store)
    try:
        if not os.path.exists(VECTORSTORE_PATH):
            print(f"❌ خطأ: مجلد قاعدة البيانات المتجهة '{VECTORSTORE_PATH}' غير موجود.")
        else:
            dummy_embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=dummy_embeddings,
                persist_directory=VECTORSTORE_PATH,
            )
            vector_store_count = vectorstore._collection.count()
            print(f"✅ عدد المتجهات في قاعدة البيانات (Vector Store): {vector_store_count}")
            if vector_store_count > total_expected_chunks:
                 print("   > 👍 منطقي! عدد المتجهات أكبر بسبب التقطيع.")
            else:
                 print("   > ⚠️ قد يكون هناك نقص في المتجهات.")

    except Exception as e:
        print(f"❌ فشل في الوصول إلى قاعدة البيانات المتجهة: {e}")

    print("--- انتهت عملية التحقق ---")

if __name__ == "__main__":
    verify_databases()

