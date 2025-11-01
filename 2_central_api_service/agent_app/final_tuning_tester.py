# المسار: 2_central_api_service/agent_app/final_tuning_tester.py
# --- الإصدار 13.0: الاختبار النهائي للشخصية، الذكاء الحواري، والموثوقية ---

import os
import logging
import asyncio
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from flashrank import Ranker, RerankRequest

# --- 1. الإعدادات ---
# (نفس الإعدادات السابقة)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# --- 2. الملفات الشخصية للأنظمة (تبقى كما هي) ---
SYSTEM_PROFILES = {
    "sys": {
        "name": "نظام إدارة طلبات الاعتماد",
        "description": "نظام إلكتروني لتتبع رحلة الحصول على الاعتماد.",
        "keywords": ["إنشاء حساب", "تسجيل الدخول", "طلب اعتماد", "قوائم التحقق", "دراسة مكتبية", "زيارة ميدانية", "إجراءات تصحيحية", "فاتورة", "شهادة"]
    },
    "university_alpha": {
        "name": "تطبيق Plant Care",
        "description": "تطبيق ذكي لتشخيص أمراض النباتات والآفات الزراعية.",
        "keywords": ["تشخيص النبات", "آفات زراعية", "متطلبات وظيفية", "حالات استخدام", "تصميم النظام", "plant care"]
    },
    "school_beta": {
        "name": "مستندات الشبكات العصبية",
        "description": "مادة تعليمية عن الشبكات العصبية و TensorFlow.",
        "keywords": ["شبكة عصبية", "tensorflow", "cnn", "layer", "relu", "pooling", "optimizer"]
    },
    "un": {
        "name": "بوابة المشتريات الإلكترونية للأمم المتحدة",
        "description": "دليل إرشادي للموردين لاستخدام نظام الشراء الإلكتروني.",
        "keywords": ["مناقصات", "تسجيل الدخول", "عطاءات", "unops", "esourcing", "ungm.org", "موردين"]
    }
}

# --- 3. القوالب النهائية (الإصدار 13.0) ---

# قالب إعادة الصياغة (يبقى كما هو من الإصدار 12.0 لأنه أثبت نجاحه)
REWRITE_PROMPT_TEMPLATE = """
مهمتك هي استخراج الكلمات المفتاحية الأكثر أهمية من سؤال المستخدم لتحسين البحث.

**سياق النظام:** {system_name}
**مصطلحات هامة:** {system_keywords}

---
**القواعد:**
1.  **إذا كان السؤال عامًا عن النظام** (مثل "ما هو هذا النظام؟")، أرجع اسم النظام فقط: `{system_name}`.
2.  **إذا كان السؤال عن خطوات أو كيفية فعل شيء** (مثل "كيف أضيف مستخدم؟")، أرجع الفعل والمفعول به: `إضافة مستخدم جديد`.
3.  **إذا كان السؤال عن تعريف مصطلح** (مثل "ماهي الشبكات العصبية؟")، أرجع المصطلح نفسه: `الشبكات العصبية`.
4.  **إذا كان السؤال خارج السياق تمامًا** (مثل "من هو ميسي؟")، أرجع السؤال الأصلي كما هو.
5.  **الناتج يجب أن يكون قصيرًا جدًا ومباشرًا.** لا تستخدم جمل كاملة.

---
**المهمة المطلوبة:**

سؤال المستخدم: {question}

الاستعلام المحسّن:
"""

# --- قالب الإجابة النهائي مع الشخصية والذكاء الحواري ---
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
**شخصيتك:** أنت "مساعد الدعم الفني لـ OpenSoft"، خبير ومتخصص في النظام التالي: {system_name}.

**مهمتك:** الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.

**قواعد صارمة:**
1.  ابدأ دائمًا إجابتك بـ "بالتأكيد! بخصوص سؤالك عن..." أو صيغة ترحيبية مشابهة.
2.  إذا كان السياق يحتوي على إجابة واضحة، قدمها بشكل منظم ومفصل في نقاط.
3.  إذا كانت المعلومات غير موجودة في السياق، يجب أن تقول **فقط**: "بحثت في قاعدة المعرفة الخاصة بنظام '{system_name}'، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال." لا تخترع أي إجابات.
4.  بعد تقديم الإجابة، اختتم دائمًا بسؤال تفاعلي مثل: "هل تود شرحًا أكثر تفصيلاً لنقطة معينة؟" أو "هل هناك أي شيء آخر يمكنني مساعدتك به؟".
5.  لا تذكر أبدًا كلمة "سياق" أو "مستندات" للمستخدم.

---
**السياق الذي يجب الاعتماد عليه:**
{context}

---
**سؤال المستخدم:** {input}

**إجابتك:**
""")


# --- 4. الدوال المساعدة ---
def _load_all_docs_from_faiss(vs: FAISS) -> List[Document]:
    return list(vs.docstore._dict.values())

def _clean_rewritten_query(raw_query: str) -> str:
    # (تبقى كما هي)
    lines = raw_query.strip().split('\n')
    for line in reversed(lines):
        cleaned_line = line.strip()
        if cleaned_line:
            if cleaned_line.startswith("الاستعلام المحسّن:"):
                return cleaned_line.replace("الاستعلام المحسّن:", "").strip()
            return cleaned_line
    return raw_query.strip()

def print_results(docs: List[Document], title: str):
    print(f"\n--- 📄 {title} (عدد: {len(docs)}) ---")
    if not docs:
        print("   -> لا توجد مستندات.")
        return
    for i, doc in enumerate(docs):
        content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:100]
        print(f"   {i+1}. [مصدر: {doc.metadata.get('source', 'N/A')}] -> \"{content_preview}...\"")

# --- 5. الدالة الرئيسية للاختبار ---
async def run_full_test_pipeline(question: str, tenant_id: str, llm: Ollama, vector_store: FAISS, reranker: Ranker):
    print("\n" + "="*80)
    print(f"🚀 بدء اختبار كامل للسؤال: '{question}' | للعميل: '{tenant_id}'")
    print("="*80)

    # --- المرحلة 0: تحميل الملف الشخصي ---
    profile = SYSTEM_PROFILES.get(tenant_id)
    if not profile:
        print(f"⚠️ تحذير: لم يتم العثور على ملف شخصي للعميل '{tenant_id}'. سيتم استخدام السؤال الأصلي.")
        effective_question = question
    else:
        print(f"✅ [1/5] تم العثور على ملف شخصي: '{profile['name']}'")
        # --- المرحلة 1: إعادة صياغة السؤال ---
        print("🧠 [2/5] بدء إعادة صياغة السؤال باستخدام النموذج اللغوي...")
        rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
        rewriter_chain = rewrite_prompt | llm | StrOutputParser()
        raw_rewritten_query = await rewriter_chain.ainvoke({
            "system_name": profile.get("name", ""),
            "system_keywords": ", ".join(profile.get("keywords", [])),
            "question": question
        })
        effective_question = _clean_rewritten_query(raw_rewritten_query)
        print(f"   -> السؤال الأصلي: '{question}'")
        print(f"   -> الاستعلام المحسّن للبحث: '{effective_question}'")

    # --- المرحلة 2: الاسترجاع الهجين ---
    print("🔍 [3/5] بدء الاسترجاع الهجين (BM25 + FAISS)...")
    all_docs = _load_all_docs_from_faiss(vector_store)
    tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
    
    if not tenant_docs:
        print(f"❌ خطأ: لا توجد مستندات لهذا العميل في قاعدة البيانات.")
        return

    bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=10)
    faiss_retriever = vector_store.as_retriever(search_kwargs={'k': 10, 'filter': {'tenant_id': tenant_id}})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    
    initial_docs = await ensemble_retriever.ainvoke(effective_question)
    print_results(initial_docs, "النتائج الأولية من البحث الهجين")

    # --- المرحلة 3: إعادة الترتيب (Reranking) ---
    print("✨ [4/5] بدء إعادة الترتيب باستخدام FlashRank...")
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(initial_docs)]
    rerank_request = RerankRequest(query=question, passages=passages)
    all_reranked_results = reranker.rerank(rerank_request)
    top_4_results = all_reranked_results[:4]
    
    original_docs_map = {doc.page_content: doc for doc in initial_docs}
    reranked_docs = [original_docs_map[res["text"]] for res in top_4_results if res["text"] in original_docs_map]
    print_results(reranked_docs, "النتائج النهائية بعد إعادة الترتيب (Top 4)")

    # --- المرحلة 4: توليد الإجابة النهائية ---
    print("✍️ [5/5] بدء توليد الإجابة النهائية بالاعتماد على النتائج المعاد ترتيبها...")
    answer_chain = FINAL_ANSWER_PROMPT | llm | StrOutputParser()
    
    final_context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    
    final_answer = await answer_chain.ainvoke({
        "system_name": profile.get("name", "هذا النظام"),
        "context": final_context,
        "input": question
    })

    print("\n" + "-"*30 + " 💬 الإجابة النهائية 💬 " + "-"*30)
    print(final_answer)
    print("="*80)


async def main():
    print("--- 🔬 بدء تهيئة بيئة الاختبار النهائية 🔬 ---")
    try:
        llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        reranker = Ranker()
        print("--- ✅ بيئة الاختبار النهائية جاهزة ---")
    except Exception as e:
        print(f"❌ فشل فادح في التهيئة: {e}")
        return

    # --- هنا يمكنك إضافة كل حالات الاختبار التي تريدها ---
    
    # اختبار 1: سؤال عام عن نظام الاعتماد
    await run_full_test_pipeline("ماهو هذا النظام ومن يتبعه", "sys", llm, vector_store, reranker)
    
    # اختبار 2: سؤال فني محدد عن نظام الشبكات العصبية
    await run_full_test_pipeline("ماهي الشبكات العصبيه", "school_beta", llm, vector_store, reranker)

    # اختبار 3: سؤال عن خطوات في نظام الأمم المتحدة
    await run_full_test_pipeline("كيف نسجل الدخول الى النظام", "un", llm, vector_store, reranker)

    # اختبار 4: سؤال خارج السياق تمامًا
    await run_full_test_pipeline("من هي جورجينا", "sys", llm, vector_store, reranker)
    
    # اختبار 5: سؤال "من أنت؟"
    await run_full_test_pipeline("من انت", "university_alpha", llm, vector_store, reranker)


if __name__ == "__main__":
    asyncio.run(main())
