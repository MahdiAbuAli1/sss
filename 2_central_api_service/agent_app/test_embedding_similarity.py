import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv

# --- الإعدادات ---
load_dotenv() # تأكد من وجود ملف .env يحتوي على OLLAMA_HOST
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# --- دالة حساب تشابه الكوساين ---
def cosine_similarity(vec1, vec2):
    """يحسب تشابه الكوساين بين متجهين."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# --- جمل الاختبار ---
# جملتان متشابهتان جدًا في المعنى
sentence1 = "كيف أبدأ في استخدام النظام لتقديم طلب؟"
sentence2 = "ما هي أول خطوة لتقديم طلب جديد في النظام؟"

# جملة مختلفة تمامًا في المعنى
sentence3 = "ما هي عاصمة فرنسا؟"

# --- تهيئة نموذج التضمين ---
print(f"--- 🔬 بدء اختبار نموذج التضمين: {EMBEDDING_MODEL} ---")
try:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
except Exception as e:
    print(f"❌ فشل في تهيئة النموذج: {e}")
    exit()

# --- توليد المتجهات (Embeddings) ---
print("🧠 توليد المتجهات للجمل...")
try:
    vec1 = embeddings.embed_query(sentence1)
    vec2 = embeddings.embed_query(sentence2)
    vec3 = embeddings.embed_query(sentence3)
    print("✅ تم توليد المتجهات بنجاح.")
except Exception as e:
    print(f"❌ فشل في توليد المتجهات: {e}")
    exit()

# --- حساب وطباعة درجات التشابه ---
print("\n" + "="*50)
print("📊 حساب درجات التشابه (Cosine Similarity)")
print("="*50)

similarity_1_2 = cosine_similarity(vec1, vec2)
similarity_1_3 = cosine_similarity(vec1, vec3)

print(f"الجملة 1: '{sentence1}'")
print(f"الجملة 2: '{sentence2}'")
print(f"الجملة 3: '{sentence3}'")
print("-" * 50)

print(f"🎯 درجة التشابه بين الجملة 1 و 2 (المتشابهتين): {similarity_1_2:.4f}")
print(f"🎯 درجة التشابه بين الجملة 1 و 3 (المختلفتين):  {similarity_1_3:.4f}")
print("="*50)

# --- التحليل النهائي ---
print("\n--- 🕵️‍♂️ التحليل ---")
if similarity_1_2 > 0.75:
    print("✔️ نتيجة إيجابية: النموذج يميز بين الجمل المتشابهة بشكل جيد.")
else:
    print("❌ نتيجة سلبية: النموذج فشل في التعرف على تشابه الجملتين 1 و 2. يجب أن تكون الدرجة أعلى بكثير.")

if similarity_1_3 < 0.4:
    print("✔️ نتيجة إيجابية: النموذج يميز بين الجمل المختلفة بشكل جيد.")
else:
    print("❌ نتيجة سلبية: النموذج يخلط بين الجمل المختلفة. يجب أن تكون الدرجة أقل بكثير.")

if similarity_1_2 < similarity_1_3:
    print("\n🚨🚨🚨 فشل كارثي: درجة تشابه الجمل المختلفة أعلى من درجة تشابه الجمل المتشابهة! النموذج لا يعمل بشكل صحيح للغة العربية.")

