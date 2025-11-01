# المسار: 2_central_api_service/agent_app/performance_tracker.py

import time
import json
import os
import logging

# -----------------------------------------------------------------------------
# ⚙️ إعداد المسار الثابت لملف السجل
# -----------------------------------------------------------------------------
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(APP_DIR, "performance_log.jsonl")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
except Exception as e:
    logging.error(f"فشل في تحديد مسار ملف السجل: {e}")
    LOG_FILE = "performance_log.jsonl"


class PerformanceLogger:
    """
    مسجل أداء محسن يقوم بكتابة السجلات بصيغة JSON Lines.
    يستخدم مفتاحًا فريدًا لكل طلب لتجنب التضارب في البيئات المتزامنة.
    """
    def __init__(self):
        self.start_times = {}

    def start(self, stage_name: str, tenant_id: str, question: str, extra_info: dict = None) -> str:
        """
        تبدأ توقيت مرحلة معينة وتسجل معلومات البدء.
        
        Returns:
            str: مفتاح فريد للطلب لاستخدامه في دالة end.
        """
        # نستخدم مفتاحًا فريدًا لكل طلب لمنع التضارب
        request_key = f"{stage_name}-{tenant_id}-{id(question)}"
        self.start_times[request_key] = time.time()
        
        log_message = f"⏱️ بدء المرحلة: {stage_name} | العميل: {tenant_id}"
        if extra_info:
            log_message += f" | معلومات إضافية: {extra_info}"
        
        logging.info(log_message)
        return request_key # نعيد المفتاح الفريد لاستخدامه في دالة end

    def end(self, request_key: str, tenant_id: str, question: str, extra_info: dict = None):
        """
        تسجل انتهاء مرحلة معينة باستخدام المفتاح الفريد الخاص بها.
        """
        if request_key not in self.start_times:
            logging.warning(f"⚠️ محاولة إنهاء مرحلة '{request_key}' لم يتم بدؤها أو تم إنهاؤها بالفعل.")
            return

        end_time = time.time()
        duration = end_time - self.start_times[request_key]
        stage_name = request_key.split('-')[0] # نستخرج اسم المرحلة من المفتاح
        
        record = {
            "timestamp": end_time,
            "tenant_id": tenant_id,
            "question": question,
            "stage": stage_name,
            "duration_sec": round(duration, 4),
        }
        
        if extra_info:
            record.update(extra_info)
        
        try:
            # استخدام وضع الإلحاق "a" للكتابة الآمنة والفعالة
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            logging.info(f"✅ تم تسجيل المرحلة: {stage_name} | المدة: {record['duration_sec']} ثانية")

        except Exception as e:
            logging.error(f"❌ فشل في كتابة سجل الأداء للمرحلة '{stage_name}': {e}", exc_info=True)
        
        # حذف وقت البدء بعد التسجيل لمنع إعادة استخدامه بالخطأ
        del self.start_times[request_key]

