# --- مختبر المعايير الشامل: مقارنة أسس الاسترجاع وتقنياته ---

import os
import time
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- 1. استيراد المكونات ---
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from sentence_transformers.cross_encoder import CrossEncoder

load_dotenv()

# --- 2. الإعدادات العامة ---
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "4_client_docs")
TOP_K = 7
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434" )
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- 3. تعريف الأسس (Foundations) ---
FOUNDATIONS = {
    "A_Fast_Compact": {
        "name": "الأساس أ: سرعة وأصغر حجمًا",
        "splitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100),
        "embedding_model": OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b"),
            base_url=OLLAMA_BASE_URL
        ),
    },
    "B_Accurate_Contextual": {
        "name": "الأساس ب: دقة وسياق",
        "splitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        "embedding_model": HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        ),
    }
}

# --- 4. دوال مساعدة ---
def load_all_documents(path: str) -> List[Document]:
    """تحميل جميع المستندات من جميع العملاء."""
    print("--- ⏳ جارٍ تحميل جميع مستندات العملاء... ---")
    all_docs = []
    for tenant_id in os.listdir(path):
        tenant_path = os.path.join(path, tenant_id)
        if os.path.isdir(tenant_path):
            # استخدام DirectoryLoader مع أنواع متعددة من الملفات
            pdf_loader = DirectoryLoader(tenant_path, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
            docx_loader = DirectoryLoader(tenant_path, glob="**/*.docx", loader_cls=Docx2txtLoader, recursive=True)
            txt_loader = DirectoryLoader(tenant_path, glob="**/*.txt", loader_cls=TextLoader, recursive=True)
            
            docs = pdf_loader.load() + docx_loader.load() + txt_loader.load()
            
            # إضافة tenant_id إلى الميتا-بيانات
            for doc in docs:
                doc.metadata['tenant_id'] = tenant_id
            all_docs.extend(docs)
            print(f"   -> تم تحميل {len(docs)} مستند للعميل: {tenant_id}")
    return all_docs

def print_benchmark_results(title: str, docs: List[Document], duration: float, scores: List[float] = None):
    """طباعة نتائج الاختبار بشكل منظم."""
    print("\n" + "-"*40)
    print(f"🔬 {title}")
    print(f"⏱️ الزمن: {duration:.4f} ثانية | 📄 النتائج: {len(docs)}")
    print("-"*40)
    if not docs:
        print("   -> لم يتم العثور على نتائج.")
        return
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get('source', 'N/A'))
        content_preview = ' '.join(doc.page_content.replace('\n', ' ').split())[:90]
        score_info = f"[الدرجة: {scores[i]:.4f}]" if scores and i < len(scores) else ""
        print(f"   {i+1}. {score_info} [{source}] -> \"{content_preview}...\"")

# --- 5. المختبر الرئيسي ---
async def run_benchmark(question: str, tenant_id: str, foundation_builds: Dict, reranker: CrossEncoder):
    """تشغيل جميع الاختبارات لسؤال واحد."""
    print("\n" + "#"*30 + f" بدء الاختبار للعميل: '{tenant_id}' | السؤال: '{question}' " + "#"*30)

    for key, build in foundation_builds.items():
        print(f"\n{'='*20} استخدام [{build['name']}] {'='*20}")
        
        retrievers = build['retrievers'][tenant_id]
        faiss_retriever = retrievers['faiss']
        bm25_retriever = retrievers['bm25']
        ensemble_retriever = retrievers['ensemble']
        parent_document_retriever = retrievers['parent']

        # --- التقنية 1: Vector Search ---
        start_time = time.time()
        vector_docs = await faiss_retriever.ainvoke(question)
        duration = time.time() - start_time
        print_benchmark_results(f"[{build['name']}] البحث الدلالي (Vector)", vector_docs, duration)

        # --- التقنية 2: Hybrid Search ---
        start_time = time.time()
        hybrid_docs = await ensemble_retriever.ainvoke(question)
        duration = time.time() - start_time
        print_benchmark_results(f"[{build['name']}] البحث الهجين (Hybrid)", hybrid_docs, duration)

        # --- التقنية 3: Parent Document ---
        start_time = time.time()
        parent_docs = await asyncio.to_thread(parent_document_retriever.invoke, question)
        duration = time.time() - start_time
        print_benchmark_results(f"[{build['name']}] مسترجع المستند الأصل (Parent)", parent_docs, duration)

        # --- التقنية 4: Ultimate Retriever ---
        combined_docs = list({doc.page_content: doc for doc in reversed(hybrid_docs + parent_docs)}.values())[::-1]
        if combined_docs:
            start_time = time.time()
            passages = [[question, doc.page_content] for doc in combined_docs]
            scores = reranker.predict(passages)
            duration = time.time() - start_time
            
            reranked_results = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, score in reranked_results][:TOP_K]
            final_scores = [score for doc, score in reranked_results][:TOP_K]
            print_benchmark_results(f"[{build['name']}] المسترجع الشامل (Ultimate)", final_docs, duration, scores=final_scores)

# --- 6. الدالة الرئيسية للتنفيذ ---
async def main():
    print("--- 🚀 بدء مختبر المعايير الشامل (v3.0) 🚀 ---")
    
    all_docs = load_all_documents(DOCS_PATH)
    reranker = CrossEncoder(RERANK_MODEL)
    
    foundation_builds = {}

    for key, config in FOUNDATIONS.items():
        print(f"\n--- 🏗️ جارٍ بناء البنية التحتية لـ [{config['name']}] ---")
        
        # 1. التقطيع
        chunks = config['splitter'].split_documents(all_docs)
        
        # 2. بناء قاعدة البيانات المتجهة
        print(f"   -> جارٍ بناء قاعدة بيانات FAISS مع نموذج {config['embedding_model'].__class__.__name__}...")
        vector_store = await asyncio.to_thread(FAISS.from_documents, chunks, config['embedding_model'])
        
        # 3. تجميع المستندات والمقاطع لكل عميل
        tenant_chunks = {}
        for chunk in chunks:
            tenant_id = chunk.metadata['tenant_id']
            if tenant_id not in tenant_chunks:
                tenant_chunks[tenant_id] = []
            tenant_chunks[tenant_id].append(chunk)
            
        # 4. تهيئة المسترجعات لكل عميل
        tenant_retrievers = {}
        for tenant_id, t_chunks in tenant_chunks.items():
            faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
            bm25_retriever = BM25Retriever.from_documents(t_chunks, k=TOP_K)
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
            
            # Parent retriever يحتاج إلى المقاطع الأصلية
            original_tenant_docs = [doc for doc in all_docs if doc.metadata['tenant_id'] == tenant_id]
            store = InMemoryStore()
            parent_document_retriever = ParentDocumentRetriever(
                vectorstore=vector_store, 
                docstore=store, 
                child_splitter=config['splitter'],
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000) # مقاطع أصلية أكبر
            )
            parent_document_retriever.add_documents(original_tenant_docs, ids=None)
            
            tenant_retrievers[tenant_id] = {
                'faiss': faiss_retriever,
                'bm25': bm25_retriever,
                'ensemble': ensemble_retriever,
                'parent': parent_document_retriever
            }
        
        foundation_builds[key] = {
            "name": config["name"],
            "retrievers": tenant_retrievers
        }
        print(f"--- ✅ تم بناء البنية التحتية لـ [{config['name']}] بنجاح ---")

    # --- تعريف حالات الاختبار ---
    test_cases = [
        {"tenant_id": "school_beta", "question": "ما هي مكتبة TensorFlow؟"},
        {"tenant_id": "school_beta", "question": "قارن بين الطبقة التلافيفية والطبقة الكثيفة."},
        {"tenant_id": "sys", "question": "ماذا يحدث بعد سداد الفاتورة المبدئية في رحلة الحصول على الاعتماد؟"},
        {"tenant_id": "school_beta", "question": "كيف يمكن مواجهة مشكلة تلاشي مشتقة الخطأ (Vanishing Gradient)؟"},
        {"tenant_id": "university_alpha", "question": "ما هي الفائدة الاقتصادية لتطبيق Plant Care للمزارعين، وما هي حدوده الوظيفية؟"},
        {"tenant_id": "un", "question": "ماذا يحدث بعد تقديم العطاء وقبل إرساء العقد؟"}
    ]

    # --- تشغيل جميع الاختبارات ---
    for case in test_cases:
        await run_benchmark(case["question"], case["tenant_id"], foundation_builds, reranker)

    print("\n--- 🎉 انتهى مختبر المعايير الشامل. ---")

if __name__ == "__main__":
    asyncio.run(main())
