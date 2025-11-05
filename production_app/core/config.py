# production_app/core/config.py

import os
import logging
from logging.handlers import RotatingFileHandler

# --- المسارات الأساسية ---
# تحديد المسار الجذري للمشروع بشكل آمن
# --- المسارات الأساسية ---
# المسار إلى مجلد التطبيق الحالي (production_app)
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# المسار إلى المجلد الجذري للمشروع بأكمله (support_service_platform)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

STATIC_DIR = os.path.join(APP_ROOT, "static")
LOGS_DIR = os.path.join(APP_ROOT, "logs")

# تأكد من وجود مجلد السجلات
os.makedirs(LOGS_DIR, exist_ok=True)
# تأكد من وجود مجلد السجلات
os.makedirs(LOGS_DIR, exist_ok=True)

# --- إعدادات النماذج ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen3:1.7b.7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST","http://127.0.0.1:11434")

# --- إعدادات RAG ---
TOP_K = 7
RELEVANCE_THRESHOLD = 0.3 # درجة الصلة الدنيا لقبول المستندات
MIN_QUESTION_LENGTH = 3

# --- إعدادات ذاكرة المحادثة ---
# ذاكرة بسيطة تعتمد على معرّف الجلسة، يتم الاحتفاظ بها في الذاكرة RAM
# للإنتاج الفعلي على نطاق واسع، يمكن استبدالها بـ Redis أو قاعدة بيانات
SESSION_MEMORY = {}

# --- مسارات قواعد البيانات ---
# افترض أن هذه المجلدات موجودة خارج مجلد production_app
# مثال: /your_main_folder/3_shared_resources/
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
HIERARCHICAL_DB_PATH = os.path.join(PROJECT_ROOT, "2_central_api_service", "agent_app", "hierarchical_db.json")
# --- إعدادات التسجيل (Logging) للإنتاج ---
def setup_logging():
    # سجل التطبيق العام
    app_log_path = os.path.join(LOGS_DIR, "app.log")
    app_handler = RotatingFileHandler(app_log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    app_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
    app_handler.setFormatter(app_formatter)
    
    # سجل التحليل
    analysis_log_path = os.path.join(LOGS_DIR, "analysis.jsonl")
    analysis_handler = logging.FileHandler(analysis_log_path, mode='a', encoding='utf-8')
    analysis_formatter = logging.Formatter('%(message)s')
    analysis_handler.setFormatter(analysis_formatter)

    # إعداد الـ loggers
    logging.basicConfig(level=logging.INFO, handlers=[app_handler, logging.StreamHandler()])
    
    analysis_logger = logging.getLogger('AnalysisLogger')
    analysis_logger.setLevel(logging.INFO)
    analysis_logger.propagate = False
    if not analysis_logger.handlers:
        analysis_logger.addHandler(analysis_handler)
        
    return analysis_logger

ANALYSIS_LOGGER = setup_logging()
