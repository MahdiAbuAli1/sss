# test_retrieval.py
import os
from project_core.core.retrieval import retriever

def test_retrieval_relevance():
    print("="*30 + "\n🔬 بدء اختبار جودة الاسترجاع 🔬\n" + "="*30)

    if not retriever:
        print("❌ خطأ: فشل تهيئة المسترجع (Retriever). يرجى التحقق من ملفات config و main.")
        return

    while True:
        try:
            question = input("\n> أدخل سؤالك للاختبار (أو اكتب 'خروج' للإنهاء): ")
            if question.lower() in ['خروج', 'exit', 'quit']:
                break
            
            tenant_id = input("> أدخل tenant_id (مثال: perfume_shop_01): ")
            if not tenant_id:
                print("Tenant ID مطلوب.")
                continue

            print("\n--- البحث عن المستندات ذات الصلة... ---")
            
            # استخدام المسترجع مع الفلتر
            session_retriever = retriever.vectorstore.as_retriever(
                search_kwargs={'k': 3, 'filter': {'tenant_id': tenant_id}}
            )
            
            # استخدام الدالة الأحدث .invoke()
            docs = session_retriever.invoke(question)

            if not docs:
                print("\n⚠️ لم يتم العثور على أي مستندات ذات صلة.")
                continue

            print(f"\n✅ تم العثور على {len(docs)} مستندات. إليك تحليلها:\n")
            for i, doc in enumerate(docs):
                print(f"--- المستند رقم {i+1} ---")
                print(f"  - النوع: {doc.metadata.get('type')}")
                print(f"  - المصدر: {doc.metadata.get('source_file')}")
                print(f"  - المحتوى المُثرى:\n{doc.page_content}\n")

        except Exception as e:
            print(f"\n❌ حدث خطأ أثناء الاختبار: {e}")

    print("\n" + "="*30 + "\n✅ اكتمل اختبار جودة الاسترجاع.\n" + "="*30)

if __name__ == "__main__":
    test_retrieval_relevance()
