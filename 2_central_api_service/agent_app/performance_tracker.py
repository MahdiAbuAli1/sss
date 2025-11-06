# المسار: 2_central_api_service/agent_app/performance_tracker.py
# الإصدار: v11.1 - The Analyst's Toolkit (Final Fix)

import os
import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict
from langchain_core.documents import Document
import json  # <--- هذا هو السطر الذي تم إضافته
import asyncio # <--- إضافة هذا السطر لتعريف القفل بشكل صحيح

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../5_analysis_logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

class RequestLogger:
    LOG_FILE_PATH = os.path.join(LOGS_DIR, "rag_analysis_log.jsonl")
    # استخدام قفل asyncio لضمان الكتابة الآمنة من عمليات غير متزامنة متعددة
    _lock = asyncio.Lock()

    def __init__(self, session_id: str, question: str):
        self.log_data = {
            "request_id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "question": question,
            "stages": [],
            "final_answer": None,
            "total_duration_seconds": 0
        }
        self.start_time = time.time()

    def add_stage(self, name: str, duration: float, details: Dict):
        self.log_data["stages"].append({
            "name": name,
            "duration_seconds": round(duration, 4),
            "details": details
        })

    def set_final_answer(self, answer: str):
        self.log_data["final_answer"] = answer

    async def save(self):
        self.log_data["total_duration_seconds"] = round(time.time() - self.start_time, 4)
        # استخدام json.dumps الذي تم استيراده الآن
        log_json_string = json.dumps(self.log_data, ensure_ascii=False)
        
        async with self._lock:
            try:
                # استخدام الكتابة غير المتزامنة للكفاءة (اختياري ولكن أفضل)
                with open(self.LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write(log_json_string + "\n")
            except IOError as e:
                logging.error(f"❌ فشل الكتابة إلى ملف السجل المركزي: {e}")

def format_docs_for_logging(docs: List[Document]) -> List[Dict]:
    if not docs:
        return []
    
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'N/A').split(os.sep)[-1]
        content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:120] + "..."
        formatted.append({"source": source, "content_preview": content_preview})
    return formatted
