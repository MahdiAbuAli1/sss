# project_core/core/config.py

import os
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# --- المسارات الأساسية للمشروع ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- تعريف المجلدات الرئيسية ---
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate_outputs")
ENRICHED_DIR = os.path.join(BASE_DIR, "enriched_outputs")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_outputs")

# --- إعدادات مخازن البيانات ---
VECTORSTORE_PATH = os.path.join(STORAGE_DIR, "arabic_multimodal_vector_db")
DOCSTORE_PATH = os.path.join(STORAGE_DIR, "arabic_doc_store")
COLLECTION_NAME = "arabic_multimodal_collection"
PROCESSED_LOG_FILE = os.path.join(STORAGE_DIR, "processed_files.log")

# --- إعدادات المعالجة ---
ENABLE_PROCESSING_EXTRACTION = True
ENABLE_PROCESSING_ENRICHMENT = True

# --- تحديد عنوان خادم Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- تهيئة نماذج اللغة المحلية العربية عبر Ollama ---

def get_multimodal_llm( ):
    """
    يوفر نموذجًا متعدد الوسائط لفهم الصور.
    """
    return ChatOllama(model="qwen2.5vl:7b", base_url=OLLAMA_BASE_URL)

def get_text_enrichment_llm():
    """
    يوفر نموذجًا لغويًا لإثراء وتنظيم النصوص والجداول.
    """
    return ChatOllama(model="qwen2:7b-instruct-q3_K_M", base_url=OLLAMA_BASE_URL)

def get_embeddings_model():
    """
    يوفر نموذج التضمين لتحويل النصوص إلى متجهات.
    """
    return OllamaEmbeddings(model="qwen3-embedding:4b", base_url=OLLAMA_BASE_URL)

def get_generative_llm():
    """
    يوفر النموذج اللغوي الكبير لتوليد الإجابات النهائية.
    """
    # --- **التعديل هنا: استخدام النموذج الذي تملكه بالفعل** ---
    return ChatOllama(model="qwen2:7b-instruct-q3_K_M", base_url=OLLAMA_BASE_URL)
