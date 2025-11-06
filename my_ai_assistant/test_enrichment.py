# test_enrichment.py
import os
import json
import random

ENRICHED_DIR = "enriched_outputs"

def test_enrichment_quality(sample_size=3):
    print("="*30 + "\n🔬 بدء اختبار جودة الإثراء 🔬\n" + "="*30)
    
    if not os.path.exists(ENRICHED_DIR):
        print(f"❌ خطأ: مجلد '{ENRICHED_DIR}' غير موجود. يرجى تشغيل مرحلة الإثراء أولاً.")
        return

    all_files = [f for f in os.listdir(ENRICHED_DIR) if f.endswith(".json") and not f.startswith("temp_")]
    if not all_files:
        print("⚠️ لا توجد ملفات مُثراة لاختبارها.")
        return

    print(f"🔍 تم العثور على {len(all_files)} ملفات مُثراة. سيتم أخذ عينات منها...\n")

    for filename in all_files:
        print(f"\n--- تحليل الملف: {filename} ---\n")
        file_path = os.path.join(ENRICHED_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # فصل القطع حسب النوع
        texts = [c for c in chunks if c['type'] == 'text']
        tables = [c for c in chunks if c['type'] == 'table']
        images = [c for c in chunks if c['type'] == 'image']

        print(f"  - نصوص: {len(texts)} | جداول: {len(tables)} | صور: {len(images)}")

        # اختبار عينات من كل نوع
        if texts:
            print("\n  --- عينة إثراء نص:")
            sample = random.choice(texts)
            print(f"    [الأصلي]: {sample['original_content'][:150]}...")
            print(f"    [المُثرى]: {sample['enriched_content']}")
        
        if tables:
            print("\n  --- عينة إثراء جدول:")
            sample = random.choice(tables)
            print(f"    [الأصلي]: {sample['original_content'][:150]}...")
            print(f"    [المُثرى]: {sample['enriched_content']}")

        if images:
            print("\n  --- عينة إثراء صورة:")
            sample = random.choice(images)
            # لا نطبع المحتوى الأصلي (base64) لأنه طويل جدًا
            print(f"    [المُثرى]: {sample['enriched_content']}")

    print("\n" + "="*30 + "\n✅ اكتمل اختبار جودة الإثراء.\n" + "="*30)

if __name__ == "__main__":
    test_enrichment_quality()
