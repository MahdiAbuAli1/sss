# المسار: 2_central_api_service/agent_app/instant_response_tester.py
# --- مختبر اختبار طبقة الردود الفورية ---

import os
import json
import random
import time
from typing import Dict, List, Optional

# --- 1. فئة الردود الفورية (نفس الفئة التي سنستخدمها في الإنتاج) ---
class InstantResponder:
    def __init__(self, responses_dir: str):
        self.responses: Dict[str, List[str]] = {}
        self.responses_dir = responses_dir
        self.load_all_responses()

    def load_all_responses(self):
        """يقوم بتحميل جميع ملفات JSON للردود الجاهزة من المجلد المحدد."""
        print(f"🧠 تحميل الردود الجاهزة من المجلد: '{self.responses_dir}'...")
        if not os.path.isdir(self.responses_dir):
            print(f"❌ خطأ: مجلد الردود الجاهزة '{self.responses_dir}' غير موجود.")
            return

        file_count = 0
        qa_count = 0
        for filename in os.listdir(self.responses_dir):
            if filename.endswith(".json"):
                file_count += 1
                try:
                    with open(os.path.join(self.responses_dir, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            question = item.get("question")
                            answers = item.get("answers")
                            if question and answers:
                                self.responses[question.lower().strip()] = answers
                                qa_count += 1
                except Exception as e:
                    print(f"   - ❌ فشل تحميل الملف '{filename}': {e}")
        
        print(f"✅ تم تحميل {qa_count} سؤال وجواب من {file_count} ملف.")

    def get_response(self, question: str) -> Optional[str]:
        """يبحث عن السؤال ويُرجع إجابة عشوائية إذا وجده."""
        normalized_question = question.lower().strip()
        if normalized_question in self.responses:
            return random.choice(self.responses[normalized_question])
        return None

# --- 2. دالة الاختبار ---
def run_test():
    """
    يشغل سلسلة من الاختبارات على نظام الردود الفورية.
    """
    # تحديد مسار مجلد الردود الجاهزة
    canned_responses_dir = os.path.join(os.path.dirname(__file__), "canned_responses")
    
    # تهيئة نظام الردود الفورية
    responder = InstantResponder(canned_responses_dir)
    
    if not responder.responses:
        print("\nلم يتم تحميل أي ردود. لا يمكن إجراء الاختبار.")
        return

    print("\n" + "="*50)
    print("🚀 بدء اختبار الردود الفورية 🚀")
    print("="*50)

    test_cases = [
        # حالات اختبار من ملفاتك
        "شكرا لك",          # from thanks_phrases.json
        "السلام عليكم",      # from compliments_conversations_dataset.json
        "أنت غبي جداً",     # from insults_and_responses.json
        "ما اسمك الحقيقي؟", # from dataset_inappropriate_responses.json
        "لللللل",           # from random_words_dataset.json
        "الله يوفقك",       # from farewell_prayers_dataset.json
        
        # حالة اختبار غير موجودة (يجب أن تفشل)
        "ما هي عاصمة اليمن؟"
    ]

    total_time = 0
    for question in test_cases:
        start_time = time.time()
        answer = responder.get_response(question)
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        total_time += duration_ms

        print(f"\n❓ السؤال: '{question}'")
        if answer:
            print(f"   -> 💬 الإجابة: '{answer}'")
            print(f"   -> ⏱️ الزمن: {duration_ms:.2f} مللي ثانية (سريع جداً!)")
        else:
            print("   -> ❌ لا يوجد رد فوري (سيتم توجيهه إلى RAG)")
    
    print("\n" + "="*50)
    print("🎉 اكتمل الاختبار 🎉")
    print(f"⚡️ متوسط زمن الاستجابة: {total_time / len(test_cases):.2f} مللي ثانية")
    print("="*50)


# --- 3. نقطة الدخول ---
if __name__ == "__main__":
    run_test()
