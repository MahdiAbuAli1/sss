# المسار: 2_central_api_service/agent_app/core_logic.py

import os
import logging
import asyncio
import httpx
from typing import AsyncGenerator, Dict
import re
import json

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from .performance_tracker import PerformanceLogger

# --- 1. الإعدادات ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:4b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
CLIENT_DOCS_PATH = os.path.join(PROJECT_ROOT, "4_client_docs")

# --- متغيرات عالمية ---
llm: Ollama = None
vector_store: FAISS = None
embeddings: OllamaEmbeddings = None
initialization_lock = asyncio.Lock()
perf_logger = PerformanceLogger()
# حالة محادثة بسيطة لكل عميل
CONVO_STATE: Dict[str, Dict[str, str]] = {}

# تحميل ملف تعريف المستفيد (اختياري)
def load_tenant_profile(tenant_id: str) -> Dict:
    try:
        cfg_path = os.path.join(CLIENT_DOCS_PATH, tenant_id, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"تعذر قراءة ملف تعريف المستفيد {tenant_id}: {e}")
    return {}

# --- 2. القالب البسيط والنهائي ---
SIMPLE_PROMPT_TEMPLATE = """أنت مساعد دعم فني. أجب باستخدام المعلومات أدناه فقط.
إذا كانت المعلومات غير كافية تمامًا، اكتب: "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال." ولا تضف أي افتراضات.

السياق:
{context}

السؤال:
{question}

اكتب الإجابة بصيغة منظمة وواضحة كالتالي:
1) ملخص مختصر
2) الخطوات التفصيلية
3) ملاحظات/تنبيهات (إن وجدت)
4) المصادر (اذكر أسماء الملفات أو المسارات فقط مما في السياق)
"""

SIMPLE_PROMPT = PromptTemplate(template=SIMPLE_PROMPT_TEMPLATE, input_variables=["context", "question"])

# --- 3. الدوال الأساسية ---
async def initialize_agent():
    global llm, embeddings, vector_store
    async with initialization_lock:
        if vector_store is not None:
            return
        logging.info("بدء تهيئة النماذج وقاعدة البيانات الموحدة...")
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=10.0)
            llm = Ollama(
                model=CHAT_MODEL,
                base_url=OLLAMA_HOST,
                temperature=0.0,
                num_predict=384,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
            if not os.path.isdir(UNIFIED_DB_PATH):
                raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة. يرجى تشغيل 'main_builder.py' أولاً.")

            vector_store = await asyncio.to_thread(
                FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            logging.info("✅ الوكيل جاهز للعمل بالبنية البسيطة والفعالة.")
        except Exception as e:
            logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
            raise

# --- 4. دالة get_answer_stream ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    tenant_id = request_info.get("tenant_id")
    k_results = int(request_info.get("k_results", 10) or 10)

    if not vector_store:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
        return
    
    if not tenant_id:
        yield {"type": "error", "content": "معرف العميل (tenant_id) مفقود."}
        return

    logging.info(f"[{tenant_id}] بدء معالجة السؤال: '{question}'")

    # توحيد وتهيئة النص العربي مع بعض تحمل الأخطاء الإملائية البسيطة
    def normalize_ar(s: str) -> str:
        s = s.strip()
        # إزالة التشكيل
        s = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", s)
        # توحيد الألف
        s = re.sub(r"[أإآٱ]", "ا", s)
        # توحيد الياء والألف المقصورة
        s = s.replace("ى", "ي").replace("ئ", "ي").replace("ٮ", "ب")
        # همزات شائعة
        s = s.replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
        # مسافات إضافية
        s = re.sub(r"\s+", " ", s)
        return s

    q_norm = normalize_ar(question.replace("؟", ""))

    # فلترة الأسئلة القصيرة وغير الواضحة + ألفاظ إشارية تحتاج توضيح
    DEICTIC = ["هذا", "هذه", "ذلك", "تلك", "هي", "هو"]
    if len(q_norm) <= 3 or q_norm in DEICTIC:
        yield {"type": "chunk", "content": "يرجى توضيح سؤالك ضمن نطاق النظام (مثال: كيفية التسجيل/الدخول/الرفع/المشاكل)."}
        return

    # حارس أسئلة خارج النطاق الشائع (سياسة ثابتة)
    OOD_PATTERNS = [
        "من انت", "من انتم", "من تكون", "من تكونون",
        "ماهو الطب", "ما هو الطب", "ما هو الذكاء الاصطناعي", "من هو ميسي", "من هو رونالدو",
    ]
    if any(pat in q_norm for pat in OOD_PATTERNS):
        policy_msg = (
            "أنا وكيل دعم يجيب فقط من قاعدة المعرفة الخاصة بالعميل المحدد. "
            "يرجى طرح أسئلة ضمن نطاق النظام (مثل التسجيل، الدخول، رفع العروض، الأخطاء في النظام)."
        )
        yield {"type": "chunk", "content": policy_msg}
        return

    # نية: "ماهو هذا النظام" / تعريف قدرات المساعد وفق المستفيد
    WHAT_IS_PATTERNS = [
        "ماهو هذا النظام", "ما هو هذا النظام", "ماهو النظام", "ما هو النظام", "ماهو", "ما هو", "ماهي قاعده المعرفه", "ماهي قاعدة المعرفة"
    ]
    if any(pat == q_norm or (pat in q_norm and len(q_norm) <= 20) for pat in WHAT_IS_PATTERNS):
        profile = load_tenant_profile(tenant_id)
        sys_name = profile.get("system_name") or tenant_id
        capabilities = profile.get("capabilities") or []
        examples = profile.get("examples") or []
        if not capabilities:
            # محاولة ذكية لاستخلاص قدرات من أسماء الملفات
            tenant_dir = os.path.join(CLIENT_DOCS_PATH, tenant_id)
            if os.path.isdir(tenant_dir):
                files = [fn for fn in os.listdir(tenant_dir) if os.path.isfile(os.path.join(tenant_dir, fn))]
                # استنتاج بسيط: أسماء الملفات قد تعطي فكرة عامة
                capabilities = [f"المراجع المتاحة: {', '.join(files[:5])}"] if files else []
        response_lines = [
            f"اسم النظام: {sys_name}",
            "ما الذي أستطيع مساعدتك به ضمن هذا النظام:",
        ]
        if capabilities:
            response_lines += ["- " + c for c in capabilities]
        else:
            response_lines += [
                "- الإجابة عن أسئلة الاستخدام من مستندات العميل.",
                "- إرشادك لخطوات التسجيل وتسجيل الدخول إن وُجدت بالمراجع.",
                "- تلخيص وشرح المقاطع ذات الصلة من الوثائق.",
            ]
        if examples:
            response_lines += ["\nأمثلة أسئلة مقترحة:"] + ["- " + e for e in examples[:5]]
        response = "\n".join(response_lines)
        yield {"type": "chunk", "content": response}
        return

    # نوايا تفاعلية: ترحيب وشكر ومساعدة
    GREET = ["السلام عليكم", "السلام", "مرحبا", "اهلا", "أهلًا", "تحيه", "صباح الخير", "مساء الخير", "هلا", "كيف الحال", "كيفك"]
    THANKS = ["شكرا", "شكرًا", "مشكور", "يعطيك العافيه", "thx", "thanks"]
    HELP = ["مساعده", "مساعدة", "ساعدني", "اعانه", "help", "اريد الدعم الفني", "احتاج دعم فني", "اريد الدعم", "ماذا يمكنك ان تعمل", "ما الذي يمكنك فعله", "بماذا تستطيع مساعدتي"]
    FAREWELL = ["مع السلامه", "الى اللقاء", "وداعا", "باي"]

    if any(w in q_norm for w in GREET):
        yield {"type": "chunk", "content": "وعليكم السلام ورحمة الله وبركاته. أنا هنا لمساعدتك في أسئلة نظامكم. ما الذي تود القيام به؟"}
        return
    if any(w in q_norm for w in THANKS):
        yield {"type": "chunk", "content": "على الرحب والسعة! إذا احتجت أي مساعدة إضافية فأنا جاهز."}
        return
    if any(w in q_norm for w in HELP):
        profile = load_tenant_profile(tenant_id)
        caps = profile.get("capabilities") or []
        msg = "أستطيع المساعدة ضمن هذا النظام في:"
        if caps:
            msg += "\n- " + "\n- ".join(caps[:8])
        else:
            msg += "\n- الإجابة من مستندات العميل المتاحة.\n- شرح/تلخيص المقاطع ذات الصلة.\n- إرشاد خطوات التسجيل/الدخول إن كانت موثقة."
        msg += "\n\nما المطلوب تحديدًا؟"
        yield {"type": "chunk", "content": msg}
        return
    if any(w in q_norm for w in FAREWELL):
        yield {"type": "chunk", "content": "سعدت بخدمتك. في أمان الله."}
        return

    # أوضاع التقديم: اختصار/تفصيل
    summarize_mode = any(p in q_norm for p in ["اختصر", "لخص", "ملخص", "اختصر السابق", "لخص السابق", "اختصر الاجابه", "اختصر الاجابة"]) 
    detail_mode = any(p in q_norm for p in ["اشرح بالتفصيل", "بالتفصيل", "تفصيل", "شرح مفصل", "وضح بالتفصيل", "اشرح السابق", "اشرح السابق بالتفصيل", "شرح اكثر", "تفاصيل اكثر"]) 

    # إن كان الطلب يشير إلى السابق ولدينا حالة محفوظة، نتجاوز الاسترجاع ونبني على الإجابة السابقة
    if (summarize_mode or detail_mode) and tenant_id in CONVO_STATE and CONVO_STATE[tenant_id].get("last_answer"):
        prev = CONVO_STATE[tenant_id]
        prev_answer = prev.get("last_answer", "")
        prev_context = prev.get("last_context", "")
        prev_sources = prev.get("last_sources", [])
        gen_key = perf_logger.start("qa_generation", tenant_id, question, {"mode": "followup"})
        instruction = "قدم الإجابة المختصرة جدًا بنقاط معدودة وبدون تفاصيل زائدة." if summarize_mode else "قدم شرحًا تفصيليًا بخطوات واضحة وأمثلة إن وجدت."
        prompt_text = (
            "أنت مساعد دعم فني. استند فقط إلى الإجابة والسياق التاليين لتعديل الصياغة حسب الطلب.\n\n"
            f"الإجابة السابقة:\n{prev_answer}\n\n"
            f"السياق (مقتطفات):\n{prev_context[:2000]}\n\n"
            f"التعليمات: {instruction}\n"
        )
        answer = await llm.ainvoke(prompt_text)
        if prev_sources:
            answer = f"{answer}\n\nالمصادر (تجميعيًا):\n- " + "\n- ".join(prev_sources)
        perf_logger.end(gen_key, tenant_id, question, {"answer_length": len(answer or ""), "found_answer": bool(answer and answer.strip())})
        yield {"type": "chunk", "content": answer}
        # لا نحدث الحالة هنا لأنها متابعة لنفس الموضوع
        return
    try:
        total_key = perf_logger.start(
            "total_request", tenant_id, question,
            {"k_results": k_results, "vector_db_path": UNIFIED_DB_PATH, "chat_model": CHAT_MODEL, "embedding_model": EMBEDDING_MODEL}
        )

        # تفاصيل الاسترجاع مع الدرجات
        retrieval_key = perf_logger.start("retrieval", tenant_id, question, {"k": k_results})
        docs_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score, question, k_results, {"tenant_id": tenant_id}
        )
        retrieved = [
            {
                "id": (d.metadata.get("id") or d.metadata.get("source") or f"doc_{i}"),
                "score": float(score),
                "tenant_id": d.metadata.get("tenant_id"),
                "source": d.metadata.get("source"),
                "chunk": d.metadata.get("chunk_index")
            }
            for i, (d, score) in enumerate(docs_with_scores)
        ]
        scores = [float(s) for (_d, s) in docs_with_scores]
        score_min = min(scores) if scores else None
        score_mean = sum(scores)/len(scores) if scores else None
        perf_logger.end(
            retrieval_key, tenant_id, question,
            {"retrieved_count": len(retrieved), "retrieved": retrieved, "score_min": score_min, "score_mean": score_mean}
        )

        # إن لم يتم العثور على أي سياق، نعيد رسالة واضحة مباشرة
        if len(docs_with_scores) == 0:
            not_found_msg = "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."
            yield {"type": "chunk", "content": not_found_msg}
            return

        # عتبة جودة للاسترجاع (مسافة؛ الأكبر أسوأ) — تشديد القبول
        ACCEPT = True
        if score_min is not None and score_min > 1.15:
            ACCEPT = False
        if score_mean is not None and score_mean > 1.15:
            ACCEPT = False
        # سجل قرار القبول
        perf_logger.start("retrieval_decision", tenant_id, question)
        perf_logger.end("retrieval_decision", tenant_id, question, {"accepted": ACCEPT})
        if not ACCEPT:
            yield {"type": "chunk", "content": "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."}
            return

        # نعيد استخدام الوثائق المسترجعة كـ context لتجنب إعادة الاسترجاع داخل سلسلة أخرى
        top_docs = [doc for (doc, _s) in docs_with_scores[: min(len(docs_with_scores), 4)]]
        context_text = "\n\n".join([doc.page_content for doc in top_docs])
        # حد أقصى لطول السياق
        if len(context_text) > 2000:
            context_text = context_text[:2000]

        # تجهيز قائمة المصادر الموثوقة من الميتاداتا
        sources_list = []
        seen = set()
        for (doc, _s) in docs_with_scores[: min(len(docs_with_scores), 6)]:
            src = doc.metadata.get("source")
            if not src:
                continue
            base = os.path.basename(src)
            if base not in seen:
                seen.add(base)
                sources_list.append(base)

        gen_key = perf_logger.start("qa_generation", tenant_id, question)
        prompt_text = SIMPLE_PROMPT.format(context=context_text, question=question)
        if summarize_mode:
            prompt_text += "\n\nالتعليمات الإضافية: قدم الإجابة المختصرة جدًا بنقاط معدودة وبدون تفاصيل زائدة."
        elif detail_mode:
            prompt_text += "\n\nالتعليمات الإضافية: قدم شرحًا تفصيليًا بخطوات واضحة وأمثلة إن وجدت."
        answer = await llm.ainvoke(prompt_text)
        if sources_list:
            answer = f"{answer}\n\nالمصادر (تجميعيًا):\n- " + "\n- ".join(sources_list)
        found_answer = bool(answer and answer.strip())
        perf_logger.end(gen_key, tenant_id, question, {"answer_length": len(answer or ""), "found_answer": found_answer})

        logging.info(f"[{tenant_id}] الإجابة الكاملة: '{answer}'")
        yield {"type": "chunk", "content": answer}
        # تحديث حالة المحادثة لهذا العميل
        CONVO_STATE[tenant_id] = {
            "last_question": question,
            "last_answer": answer,
            "last_context": context_text,
            "last_sources": sources_list,
        }

    except Exception as e:
        logging.error(f"[{tenant_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
    finally:
        try:
            perf_logger.end(total_key, tenant_id, question)
        except Exception:
            pass

def agent_ready() -> bool:
    return vector_store is not None
