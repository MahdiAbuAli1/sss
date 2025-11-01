# المسار: 2_central_api_service/agent_app/final_tuning_tester_v2.py
# --- الإصدار 14.0: تحسين الأداء، السياق الديناميكي، والتقييم الكمي ---

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

# [تحسين] استيراد مكتبات التقييم (Ragas)
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# --- 1. الإعدادات ---
# (نفس الإعدادات السابقة)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# [تحسين] إضافة متغيرات للتحكم في التحسينات الجديدة
RERANK_SCORE_THRESHOLD = 0.1 # عتبة الثقة للسياق الديناميكي (قيمة منخفضة مبدئيًا لضمان وجود سياق)

# --- 2. الملفات الشخصية والقوالب (تبقى كما هي) ---
SYSTEM_PROFILES = {
    "sys": {"name": "نظام إدارة طلبات الاعتماد", "description": "نظام إلكتروني لتتبع رحلة الحصول على الاعتماد.", "keywords": ["إنشاء حساب", "تسجيل الدخول", "طلب اعتماد", "قوائم التحقق", "دراسة مكتبية", "زيارة ميدانية", "إجراءات تصحيحية", "فاتورة", "شهادة"]},
    "university_alpha": {"name": "تطبيق Plant Care", "description": "تطبيق ذكي لتشخيص أمراض النباتات والآفات الزراعية.", "keywords": ["تشخيص النبات", "آفات زراعية", "متطلبات وظيفية", "حالات استخدام", "تصميم النظام", "plant care"]},
    "school_beta": {"name": "مستندات الشبكات العصبية", "description": "مادة تعليمية عن الشبكات العصبية و TensorFlow.", "keywords": ["شبكة عصبية", "tensorflow", "cnn", "layer", "relu", "pooling", "optimizer"]},
    "un": {"name": "بوابة المشتريات الإلكترونية للأمم المتحدة", "description": "دليل إرشادي للموردين لاستخدام نظام الشراء الإلكتروني.", "keywords": ["مناقصات", "تسجيل الدخول", "عطاءات", "unops", "esourcing", "ungm.org", "موردين"]}
}
REWRITE_PROMPT_TEMPLATE = """... (نفس القالب السابق) ..."""
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_template("""... (نفس القالب السابق) ...""")

# --- 4. الدوال المساعدة ---
def _clean_rewritten_query(raw_query: str) -> str:
    lines = raw_query.strip().split('\n')
    for line in reversed(lines):
        cleaned_line = line.strip()
        if cleaned_line:
            if cleaned_line.startswith("الاستعلام المحسّن:"):
                return cleaned_line.replace("الاستعلام المحسّن:", "").strip()
            return cleaned_line
    return raw_query.strip()

def print_results(docs: List[Document], title: str, scores: List[float] = None):
    print(f"\n--- 📄 {title} (عدد: {len(docs)}) ---")
    if not docs:
        print("   -> لا توجد مستندات.")
        return
    for i, doc in enumerate(docs):
        content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:80]
        score_info = f"[الدرجة: {scores[i]:.4f}]" if scores else ""
        print(f"   {i+1}. {score_info} [مصدر: {doc.metadata.get('source', 'N/A')}] -> \"{content_preview}...\"")

# --- 5. الدالة الرئيسية للاختبار (النسخة المحسّنة) ---
async def run_full_test_pipeline(question: str, tenant_id: str, llm: Ollama, vector_store: FAISS, reranker: Ranker, all_docs_for_bm25: Dict[str, List[Document]]):
    print("\n" + "="*80)
    print(f"🚀 بدء اختبار كامل للسؤال: '{question}' | للعميل: '{tenant_id}'")
    print("="*80)

    # --- المرحلة 0 & 1: تحميل الملف الشخصي وإعادة صياغة السؤال ---
    profile = SYSTEM_PROFILES.get(tenant_id)
    if not profile:
        print(f"⚠️ تحذير: لم يتم العثور على ملف شخصي للعميل '{tenant_id}'.")
        effective_question = question
    else:
        print(f"✅ [1/5] تم العثور على ملف شخصي: '{profile['name']}'")
        print("🧠 [2/5] بدء إعادة صياغة السؤال...")
        # ... (نفس منطق إعادة الصياغة)
        effective_question = question # تبسيط لأغراض الاختبار، يمكنك إعادة تفعيلها

    # --- المرحلة 2: الاسترجاع الهجين المحسّن ---
    print("🔍 [3/5] بدء الاسترجاع الهجين (مع الفلترة المسبقة)...")
    
    # [تحسين] 1. الفلترة المسبقة: لا يتم تحميل كل المستندات في الذاكرة.
    # يتم استخدام الفلترة مباشرة في المسترجع.
    tenant_docs = all_docs_for_bm25.get(tenant_id)
    if not tenant_docs:
        print(f"❌ خطأ: لا توجد مستندات لهذا العميل '{tenant_id}' في قاعدة البيانات.")
        return None

    # المسترجع الأول: BM25 يعمل على المستندات المفلترة مسبقًا
    bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=10)
    
    # المسترجع الثاني: FAISS يستخدم الفلترة المدمجة
    faiss_retriever = vector_store.as_retriever(
        search_kwargs={'k': 10, 'filter': {'tenant_id': tenant_id}}
    )
    
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    
    initial_docs = await ensemble_retriever.ainvoke(effective_question)
    print_results(initial_docs, "النتائج الأولية من البحث الهجين")

    # --- المرحلة 3: إعادة الترتيب مع السياق الديناميكي ---
    print("✨ [4/5] بدء إعادة الترتيب (مع السياق الديناميكي)...")
    if not initial_docs:
        print("   -> لا توجد نتائج أولية لإعادة ترتيبها.")
        reranked_docs = []
    else:
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(initial_docs)]
        rerank_request = RerankRequest(query=question, passages=passages)
        rerank_results = reranker.rerank(rerank_request)

        # [تحسين] 2. السياق الديناميكي: فلترة النتائج بناءً على درجة الثقة.
        dynamic_top_k = [res for res in rerank_results if res["score"] >= RERANK_SCORE_THRESHOLD]
        
        original_docs_map = {doc.page_content: doc for doc in initial_docs}
        reranked_docs = [original_docs_map[res["text"]] for res in dynamic_top_k if res["text"] in original_docs_map]
        reranked_scores = [res["score"] for res in dynamic_top_k]
        print_results(reranked_docs, f"النتائج النهائية بعد إعادة الترتيب (عتبة الثقة > {RERANK_SCORE_THRESHOLD})", scores=reranked_scores)

    # --- المرحلة 4: توليد الإجابة النهائية ---
    print("✍️ [5/5] بدء توليد الإجابة النهائية...")
    answer_chain = FINAL_ANSWER_PROMPT | llm | StrOutputParser()
    
    final_context_docs = reranked_docs
    if not final_context_docs and initial_docs:
        print("   -> تحذير: لم تتجاوز أي وثيقة عتبة الثقة. سيتم استخدام أفضل نتيجة من البحث الأولي كإجراء احتياطي.")
        final_context_docs = initial_docs[:1]

    final_context_str = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])
    
    final_answer = await answer_chain.ainvoke({
        "system_name": profile.get("name", "هذا النظام"),
        "context": final_context_str,
        "input": question
    })

    print("\n" + "-"*30 + " 💬 الإجابة النهائية 💬 " + "-"*30)
    print(final_answer)
    print("="*80)

    # [تحسين] 3. تجميع البيانات للتقييم
    return {
        "question": question,
        "answer": final_answer,
        "contexts": [doc.page_content for doc in final_context_docs],
        # ground_truth هو الإجابة المثالية (يجب توفيرها يدويًا للتقييم الدقيق)
        # في هذا المثال، سنتركه فارغًا لأننا لا نملكها.
        "ground_truth": "غير متوفر" 
    }


async def main():
    print("--- 🔬 بدء تهيئة بيئة الاختبار النهائية (v2) 🔬 ---")
    try:
        # إعداد نماذج اللغة والمضمنات
        llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        
        # تحميل قاعدة البيانات المتجهة
        vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # إعداد مُعيد الترتيب
        reranker = Ranker()

        # [تحسين] تحميل المستندات مرة واحدة فقط لـ BM25
        print("   -> تحميل المستندات في الذاكرة لـ BM25 (مرة واحدة فقط)...")
        all_docs = list(vector_store.docstore._dict.values())
        all_docs_for_bm25 = {}
        for doc in all_docs:
            tenant_id = doc.metadata.get("tenant_id")
            if tenant_id:
                if tenant_id not in all_docs_for_bm25:
                    all_docs_for_bm25[tenant_id] = []
                all_docs_for_bm25[tenant_id].append(doc)
        
        print("--- ✅ بيئة الاختبار النهائية جاهزة ---")
    except Exception as e:
        print(f"❌ فشل فادح في التهيئة: {e}")
        return

    test_cases = [
        {"question": "ماهو هذا النظام ومن يتبعه", "tenant_id": "sys"},
        {"question": "ماهي الشبكات العصبيه", "tenant_id": "school_beta"},
        {"question": "كيف نسجل الدخول الى النظام", "tenant_id": "un"},
        {"question": "من هي جورجينا", "tenant_id": "sys"},
        {"question": "من انت", "tenant_id": "university_alpha"},
    ]
    
    results_for_evaluation = []
    for test in test_cases:
        result = await run_full_test_pipeline(test["question"], test["tenant_id"], llm, vector_store, reranker, all_docs_for_bm25)
        if result:
            results_for_evaluation.append(result)

    # --- مرحلة التقييم باستخدام Ragas ---
    print("\n" + "="*35 + " 📊 بدء التقييم الكمي 📊 " + "="*35)
    if not results_for_evaluation:
        print("   -> لا توجد نتائج لتقييمها.")
        return

    # تحويل النتائج إلى تنسيق مقبول من Ragas
    eval_dataset = Dataset.from_list(results_for_evaluation)
    
    # تعريف المقاييس
    # ملاحظة: answer_relevancy و context_recall تتطلبان ground_truth، لذا سيتم استبعادهما الآن.
    metrics_to_run = [
        faithfulness,      # مدى التزام الإجابة بالسياق
        context_precision, # مدى دقة السياق المسترجع بالنسبة للسؤال
    ]

    # تشغيل التقييم
    # Ragas يستخدم نماذج OpenAI بشكل افتراضي، يجب تهيئته لاستخدام Ollama
    from ragas.llms import LangchainLLM
    from ragas.embeddings import LangchainEmbeddings
    
    ragas_llm = LangchainLLM(llm=llm)
    ragas_embeddings = LangchainEmbeddings(embeddings=embeddings)
    
    score = evaluate(
        eval_dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    # عرض النتائج
    df = score.to_pandas()
    print(df)
    print("="*80)


if __name__ == "__main__":
    # ملاحظة: Ragas قد يواجه مشاكل في بيئة asyncio.
    # من الأفضل تشغيل main() بشكل متزامن إذا واجهت مشاكل.
    # لكننا سنجرب asyncio أولاً.
    asyncio.run(main())

