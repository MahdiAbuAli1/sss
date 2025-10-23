# # 2_central_api_service/agent_app/core_logic.py (النسخة الاحترافية النهائية)

# import os
# import logging
# from typing import List, Dict, Any, AsyncGenerator
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import langchain
# from langchain_core.caches import InMemoryCache

# from .performance_tracker import PerformanceLogger
# #هذا الكود الاستدعا الخاص
# perf_logger = PerformanceLogger()
# # --- تفعيل الذاكرة المؤقتة (Cache) ---
# logging.info("🚀 تفعيل الذاكرة المؤقتة (InMemoryCache) لـ LangChain...")
# langchain.llm_cache = InMemoryCache()

# # --- الإعدادات الأولية ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))


# global vector_store, llm, prompt, embeddings_model
# embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# # --- قراءة الإعدادات من متغيرات البيئة ---
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
# VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))

# # --- قالب الأسئلة المحسن ---
# RAG_PROMPT_TEMPLATE = """
# **مهمتك:** أنت مساعد دعم فني خبير ومختص. استخدم المعلومات المتوفرة في "السياق" التالي للإجابة على "سؤال المستخدم" بدقة واحترافية.
# - السياق المقدم عبارة عن مجموعة من المستندات ذات الصلة.
# - إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال." ولا تحاول اختلاق إجابة.
# - أجب دائمًا باللغة العربية.

# **السياق:**
# {context}

# **سؤال المستخدم:**
# {question}

# **الإجابة:**
# """

# # --- متغيرات عالمية ---
# vector_store = None
# llm = None
# prompt = None

# def initialize_agent():
#     """ تقوم بتحميل قاعدة المعرفة والنماذج. تُستدعى مرة واحدة عند بدء تشغيل الـ API. """
#     global vector_store, llm, prompt
    
#     if vector_store:
#         logging.info("الوكيل مُهيأ بالفعل.")
#         return

#     try:
#         logging.info("="*50)
#         logging.info("🚀 بدء تهيئة وكيل الدعم الفني...")
        
#         # 1. تحميل نموذج التضمين
#         logging.info(f"تحميل نموذج التضمين: {EMBEDDING_MODEL_NAME}...")
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

#         # 2. تحميل قاعدة بيانات المتجهات FAISS
#         logging.info(f"تحميل قاعدة المعرفة من: {VECTOR_DB_PATH}...")
#         if not os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
#             raise FileNotFoundError(f"قاعدة المعرفة (index.faiss) غير موجودة في المسار: {VECTOR_DB_PATH}. يرجى تشغيل خط أنابيب البيانات أولاً.")
        
#         vector_store = FAISS.load_local(
#             VECTOR_DB_PATH,
#             embeddings=embeddings_model,
#             allow_dangerous_deserialization=True
#         )
#         logging.info("✅ تم تحميل قاعدة المعرفة بنجاح.")

#         # 3. تحميل النموذج اللغوي الكبير للمحادثة مع إعدادات إضافية
#         logging.info(f"تحميل نموذج المحادثة: {CHAT_MODEL_NAME}...")
#         llm = Ollama(
#             model=CHAT_MODEL_NAME,
#             temperature=0.1,  # تقليل العشوائية لجعل الإجابات أكثر اتساقًا
#             # يمكنك إضافة المزيد من الإعدادات هنا مثل top_p, top_k
#         )

#         # 4. إعداد قالب الأسئلة
#         prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
#         logging.info("✅ اكتملت تهيئة وكيل الدعم الفني بنجاح!")
#         logging.info("="*50)
#     except FileNotFoundError as e:
#         logging.critical(f" فشل التهيئة: ملف قاعدة المعرفة غير موجود. {e}", exc_info=True)
#         raise
#     except Exception as e:
#         logging.critical(f" فشل فادح وغير متوقع أثناء تهيئة الوكيل: {e}", exc_info=True)
#         raise

# def format_docs_with_source(docs: List[Dict[str, Any]]) -> str:
#     """ دالة مساعدة محسنة: تنسق المستندات مع ذكر مصدرها. """
#     if not docs:
#         return "لا يوجد سياق متوفر."
    
#     sources = {doc.metadata.get('source', 'مصدر غير معروف') for doc in docs}
#     formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
#     return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"
# async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
#     """
#     تستقبل سؤالاً وهوية العميل، وتستخدم سلسلة RAG لبث الإجابة بشكل تفاعلي.
#     مع تتبع الأداء لكل مرحلة: التضمين، الاسترجاع، تنسيق المستندات، واستدعاء النموذج.
#     """
#     if not vector_store or not llm or not prompt:
#         raise RuntimeError("الوكيل غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")

#     logging.info(f"استقبال طلب بث للعميل '{tenant_id}' (k={k_results}): '{question}'")

#     try:
#         # --- مرحلة التضمين (Embedding) ---
#         perf_logger.start("embedding")
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#         question_vector = embeddings_model.embed_query(question)
#         perf_logger.end("embedding", tenant_id, question)

#         # --- مرحلة استرجاع المستندات (Retriever) ---
#         perf_logger.start("retriever")
#         retriever = vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={'k': k_results, 'filter': {'tenant_id': tenant_id}}
#         )
        
#         relevant_docs = retriever.invoke(question)

#         perf_logger.end("retriever", tenant_id, question, extra_info={"retrieved_docs": len(relevant_docs)})

#         # --- مرحلة تنسيق المستندات (Format Docs) ---
#         perf_logger.start("format_docs")
#         formatted_context = format_docs_with_source(relevant_docs)
#         perf_logger.end("format_docs", tenant_id, question, extra_info={"formatted_length": len(formatted_context)})

#         # --- مرحلة استدعاء النموذج (LLM Response) ---
#         perf_logger.start("llm_response")
#         rag_chain = (
#             RunnablePassthrough.assign(context=lambda x: relevant_docs)
#             | RunnablePassthrough.assign(context=lambda x: formatted_context)
#             | prompt
#             | llm
#         )

#         logging.info(f"جارٍ البحث عن إجابة ضمن نطاق العميل '{tenant_id}'...")

#         # --- البث التفاعلي ---
#         async for chunk in rag_chain.astream({"question": question}):
#             yield chunk

#         perf_logger.end("llm_response", tenant_id, question, extra_info={"k_results": k_results})

#     except Exception as e:
#         logging.error(f"حدث خطأ أثناء بث الإجابة للعميل '{tenant_id}': {e}", exc_info=True)
#         yield "عذرًا، حدث خطأ داخلي أثناء محاولة الإجابة على سؤالك."
#         perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})


# 22222222222222_central_api_service/agent_app/core_logic.py (نسخة محسنة لتسريع مرحلة التضمين)
# {
#   "question": "من هو مشرف هذا المشروع ومن هم الطلاب الذي عملوه وفي اي جامعه ",
#   "tenant_id": "university_alpha",
#   "k_results": 4
# }
# #"""المشرف: الدكتور وليد شاهر  
# الطلاب الذين عملوا على المشروع:  
# - عبد العزيز علي حسين القاضي  
# - مهدي محمد مهدي أبو علي  
# - علي أحمد عبد الله السعيدي  
# - فاروق حسين الغريبي  
# الجامعة: جامعة العلوم والتكنولوجيا"""
# import os
# import logging
# from typing import List, Dict, Any, AsyncGenerator
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import langchain
# from langchain_core.caches import InMemoryCache

# from .performance_tracker import PerformanceLogger

# # ------------------- تسجيل الأداء -------------------
# perf_logger = PerformanceLogger()

# # ------------------- تفعيل الذاكرة المؤقتة -------------------
# logging.info("🚀 تفعيل الذاكرة المؤقتة (InMemoryCache) لـ LangChain...")
# langchain.llm_cache = InMemoryCache()

# # ------------------- الإعدادات العامة -------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))

# # ------------------- متغيرات البيئة -------------------
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
# VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))

# # ------------------- قالب الـ Prompt -------------------
# RAG_PROMPT_TEMPLATE = """
# **مهمتك:** أنت مساعد دعم فني خبير ومختص. استخدم المعلومات المتوفرة في "السياق" التالي للإجابة على "سؤال المستخدم" بدقة واحترافية.
# - السياق المقدم عبارة عن مجموعة من المستندات ذات الصلة.
# - إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال." ولا تحاول اختلاق إجابة.
# - أجب دائمًا باللغة العربية.

# **السياق:**
# {context}

# **سؤال المستخدم:**
# {question}

# **الإجابة:**
# """

# # ------------------- المتغيرات العالمية -------------------
# vector_store = None
# llm = None
# prompt = None
# embeddings_model = None  # ✅ مضافة: للاحتفاظ بنموذج التضمين في الذاكرة

# # ==============================================================
# # 🧠 تهيئة الوكيل (تحميل الموارد مرة واحدة فقط)
# # ==============================================================
# def initialize_agent():
#     """تهيئة وكيل الدعم الفني (تحميل قاعدة المعرفة والنماذج مرة واحدة فقط)."""
#     global vector_store, llm, prompt, embeddings_model

#     if vector_store:
#         logging.info("الوكيل مُهيأ بالفعل.")
#         return

#     try:
#         logging.info("=" * 60)
#         logging.info("🚀 بدء تهيئة وكيل الدعم الفني...")

#         # 1️⃣ تحميل نموذج التضمين مرة واحدة فقط (بدلاً من كل استعلام)
#         logging.info(f"📦 تحميل نموذج التضمين: {EMBEDDING_MODEL_NAME}...")
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#         logging.info("✅ تم تحميل نموذج التضمين في الذاكرة بنجاح.")

#         # 2️⃣ تحميل قاعدة بيانات المتجهات FAISS
#         logging.info(f"📂 تحميل قاعدة المعرفة من: {VECTOR_DB_PATH}...")
#         index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
#         if not os.path.exists(index_path):
#             raise FileNotFoundError(f"قاعدة المعرفة (index.faiss) غير موجودة في المسار: {VECTOR_DB_PATH}.")
        
#         vector_store = FAISS.load_local(
#             VECTOR_DB_PATH,
#             embeddings=embeddings_model,
#             allow_dangerous_deserialization=True
#         )
#         logging.info("✅ تم تحميل قاعدة المعرفة بنجاح.")

#         # 3️⃣ تحميل نموذج المحادثة
#         logging.info(f"🧩 تحميل نموذج المحادثة: {CHAT_MODEL_NAME}...")
#         llm = Ollama(model=CHAT_MODEL_NAME, temperature=0.1)

#         # 4️⃣ إعداد القالب
#         prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

#         logging.info("✅ اكتملت تهيئة وكيل الدعم الفني بنجاح!")
#         logging.info("=" * 60)

#     except Exception as e:
#         logging.critical(f"❌ فشل أثناء التهيئة: {e}", exc_info=True)
#         raise

# # ==============================================================
# # 🔧 تنسيق المستندات
# # ==============================================================
# def format_docs_with_source(docs: List[Dict[str, Any]]) -> str:
#     """تنسق المستندات مع ذكر المصدر."""
#     if not docs:
#         return "لا يوجد سياق متوفر."
    
#     sources = {doc.metadata.get("source", "مصدر غير معروف") for doc in docs}
#     formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
#     return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

# # ==============================================================
# # 🔄 بث الإجابة
# # ==============================================================
# async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
#     """تبث الإجابة على السؤال بشكل تفاعلي باستخدام RAG."""
#     if not vector_store or not llm or not prompt or not embeddings_model:
#         raise RuntimeError("⚠️ الوكيل غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")

#     logging.info(f"🗣️ استقبال سؤال من العميل '{tenant_id}': {question}")

#     try:
#         # --- مرحلة التضمين (Embedding) ---
#         perf_logger.start("embedding")
#         question_vector = embeddings_model.embed_query(question)
#         perf_logger.end("embedding", tenant_id, question)

#         # --- مرحلة الاسترجاع ---
#         perf_logger.start("retriever")
#         retriever = vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": k_results, "filter": {"tenant_id": tenant_id}},
#         )
#         relevant_docs = retriever.invoke(question)
#         perf_logger.end("retriever", tenant_id, question, extra_info={"retrieved_docs": len(relevant_docs)})

#         # --- تنسيق المستندات ---
#         perf_logger.start("format_docs")
#         formatted_context = format_docs_with_source(relevant_docs)
#         perf_logger.end("format_docs", tenant_id, question, extra_info={"formatted_length": len(formatted_context)})

#         # --- استدعاء النموذج ---
#         perf_logger.start("llm_response")
#         rag_chain = (
#             RunnablePassthrough.assign(context=lambda x: relevant_docs)
#             | RunnablePassthrough.assign(context=lambda x: formatted_context)
#             | prompt
#             | llm
#         )

#         async for chunk in rag_chain.astream({"question": question}):
#             yield chunk

#         perf_logger.end("llm_response", tenant_id, question, extra_info={"k_results": k_results})

#     except Exception as e:
#         logging.error(f"❌ حدث خطأ أثناء بث الإجابة: {e}", exc_info=True)
#         yield "عذرًا، حدث خطأ داخلي أثناء محاولة الإجابة على سؤالك."
#         perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})
#3333333333333
# #نموذج كانت سرعته 5 دقايق ويعتبر افضل من السابق 
# import os
# import logging
# import time
# from typing import List, Dict, Any, AsyncGenerator
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import langchain
# from langchain_core.caches import InMemoryCache

# from .performance_tracker import PerformanceLogger

# # -----------------------------------------------------------------------------
# # 🧩 نظام تسجيل الأداء
# # -----------------------------------------------------------------------------
# perf_logger = PerformanceLogger()

# # -----------------------------------------------------------------------------
# # 🧠 تفعيل الذاكرة المؤقتة
# # -----------------------------------------------------------------------------
# logging.info("🚀 تفعيل الذاكرة المؤقتة (InMemoryCache) لـ LangChain...")
# langchain.llm_cache = InMemoryCache()

# # -----------------------------------------------------------------------------
# # ⚙️ الإعدادات العامة
# # -----------------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))

# # -----------------------------------------------------------------------------
# # 📦 متغيرات البيئة
# # -----------------------------------------------------------------------------
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
# VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))

# # -----------------------------------------------------------------------------
# # 🧠 قالب الـ Prompt
# # -----------------------------------------------------------------------------
# RAG_PROMPT_TEMPLATE = """
# **مهمتك:** أنت مساعد دعم فني خبير ومختص. استخدم المعلومات المتوفرة في "السياق" التالي للإجابة على "سؤال المستخدم" بدقة واحترافية.
# - السياق المقدم عبارة عن مجموعة من المستندات ذات الصلة.
# - إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال." ولا تحاول اختلاق إجابة.
# - أجب دائمًا باللغة العربية.

# **السياق:**
# {context}

# **سؤال المستخدم:**
# {question}

# **الإجابة:**
# """

# # -----------------------------------------------------------------------------
# # 🌍 المتغيرات العالمية
# # -----------------------------------------------------------------------------
# vector_store = None
# llm = None
# prompt = None
# embeddings_model = None  # ✅ نموذج التضمين يُحمّل مرة واحدة فقط

# # -----------------------------------------------------------------------------
# # 🚀 تهيئة الوكيل (تحميل الموارد مرة واحدة فقط)
# # -----------------------------------------------------------------------------
# def initialize_agent():
#     """تهيئة وكيل الدعم الفني (تحميل النماذج والبيانات مرة واحدة فقط)."""
#     global vector_store, llm, prompt, embeddings_model

#     if vector_store:
#         logging.info("✅ الوكيل مُهيأ مسبقًا.")
#         return

#     try:
#         logging.info("=" * 60)
#         logging.info("🚀 بدء تهيئة وكيل الدعم الفني...")

#         # 1️⃣ تحميل نموذج التضمين مرة واحدة
#         perf_logger.start("embedding_model_load")
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#         perf_logger.end("embedding_model_load", "system", "initialization")
#         logging.info("✅ تم تحميل نموذج التضمين في الذاكرة.")

#         # 2️⃣ تحميل قاعدة بيانات المتجهات FAISS
#         perf_logger.start("vector_db_load")
#         if not os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
#             raise FileNotFoundError(f"قاعدة المعرفة (index.faiss) غير موجودة في: {VECTOR_DB_PATH}")
#         vector_store = FAISS.load_local(
#             VECTOR_DB_PATH,
#             embeddings=embeddings_model,
#             allow_dangerous_deserialization=True
#         )
#         perf_logger.end("vector_db_load", "system", "initialization")
#         logging.info("✅ تم تحميل قاعدة المعرفة بنجاح.")

#         # 3️⃣ تحميل نموذج المحادثة (LLM)
#         perf_logger.start("chat_model_load")
#         llm = Ollama(model=CHAT_MODEL_NAME, temperature=0.1)
#         perf_logger.end("chat_model_load", "system", "initialization")
#         logging.info("✅ تم تحميل نموذج المحادثة بنجاح.")

#         # 4️⃣ إعداد القالب (Prompt)
#         prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
#         logging.info("✅ اكتملت التهيئة بنجاح!")
#         logging.info("=" * 60)

#     except Exception as e:
#         logging.critical(f"❌ فشل أثناء التهيئة: {e}", exc_info=True)
#         raise

# # -----------------------------------------------------------------------------
# # 🧾 تنسيق المستندات
# # -----------------------------------------------------------------------------
# def format_docs_with_source(docs: List[Dict[str, Any]]) -> str:
#     """تنسق المستندات المسترجعة وتضيف المصادر."""
#     if not docs:
#         return "لا يوجد سياق متوفر."
#     sources = {doc.metadata.get("source", "مصدر غير معروف") for doc in docs}
#     formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
#     return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

# # -----------------------------------------------------------------------------
# # 🧠 بث الإجابة بشكل تفاعلي (RAG Stream)
# # -----------------------------------------------------------------------------
# async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
#     """
#     بث الإجابة بشكل تفاعلي مع تسجيل الأداء لكل مرحلة.
#     """
#     if not vector_store or not llm or not prompt or not embeddings_model:
#         raise RuntimeError("⚠️ الوكيل غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")

#     logging.info(f"📩 استقبال سؤال من العميل '{tenant_id}': {question}")

#     try:
#         # ================================
#         # 1️⃣ مرحلة التضمين (Embedding)
#         # ================================
#         perf_logger.start("embedding")
#         question_vector = embeddings_model.embed_query(question)
#         perf_logger.end("embedding", tenant_id, question)

#         # ================================
#         # 2️⃣ مرحلة الاسترجاع (Retriever)
#         # ================================
#         perf_logger.start("retriever")
#         retriever = vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": k_results, "filter": {"tenant_id": tenant_id}},
#         )
#         relevant_docs = retriever.invoke(question)
#         perf_logger.end("retriever", tenant_id, question, extra_info={"retrieved_docs": len(relevant_docs)})

#         # ================================
#         # 3️⃣ مرحلة تنسيق المستندات (Formatting)
#         # ================================
#         perf_logger.start("format_docs")
#         formatted_context = format_docs_with_source(relevant_docs)
#         perf_logger.end("format_docs", tenant_id, question, extra_info={"formatted_length": len(formatted_context)})

#         # ================================
#         # 4️⃣ مرحلة استدعاء النموذج (LLM)
#         # ================================
#         perf_logger.start("llm_response")
#         rag_chain = (
#             RunnablePassthrough.assign(context=lambda x: relevant_docs)
#             | RunnablePassthrough.assign(context=lambda x: formatted_context)
#             | prompt
#             | llm
#         )

#         async for chunk in rag_chain.astream({"question": question}):
#             yield chunk

#         perf_logger.end("llm_response", tenant_id, question, extra_info={"k_results": k_results})

#     except Exception as e:
#         logging.error(f"❌ خطأ أثناء بث الإجابة: {e}", exc_info=True)
#         yield "عذرًا، حدث خطأ داخلي أثناء معالجة سؤالك."
#         perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})
 

#التعديل الجديد مع اضافه نموذج ترتيب وتعديل نموذج البحث الى نموذج بحث هجين 
# core_logic.py
# core_logic.py
# core_logic.py
# core_logic.py
# core_logic.py (النسخة النهائية - مع التوجيه والهوية الديناميكية)
# 
# # ميسي: لا يوجد أي معلومات عن ميسي في السياق المقدم.  
#الدكتور وليد شاهر: هو رئيس قسم تكنولوجيا المعلومات في جامعة العلوم والتكنولوجيا في اليمن، ويعتبر مشرفًا على مشروع هذا الكود ممتاز من حيث النتائج لكنه بطي بس لانه يستخدم النموذج اللغوي الكبير في تصنيف نوع السوال هل هوعام ام فني ام دعم وهذه المرحله تستهلك الكثبر من الوقت 
# import os
# import logging
# import time
# from typing import List, AsyncGenerator, Dict
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import langchain
# from langchain_core.caches import InMemoryCache
# from langchain_core.documents import Document
# from sentence_transformers import CrossEncoder
# from rank_bm25 import BM25Okapi

# from .performance_tracker import PerformanceLogger

# # -----------------------------------------------------------------------------
# # 🧩 إعدادات عامة وتسجيل
# # -----------------------------------------------------------------------------
# perf_logger = PerformanceLogger()
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))
# langchain.llm_cache = InMemoryCache()

# # -----------------------------------------------------------------------------
# # 📦 متغيرات البيئة والنماذج
# # -----------------------------------------------------------------------------
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
# VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))
# RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# # -----------------------------------------------------------------------------
# # 🧠 قوالب الـ Prompts (مع دعم الشخصية الديناميكية)
# # -----------------------------------------------------------------------------

# # --- 1. قالب التوجيه (Classifier) ---
# ROUTING_PROMPT_TEMPLATE = """
# مهمتك هي تصنيف سؤال المستخدم إلى أحد الفئتين التاليتين: "technical" أو "general".
# - "technical": إذا كان السؤال يتطلب البحث عن معلومات أو تفاصيل في قاعدة معرفة. (مثل: من هو المشرف، ما هو الرقم الأكاديمي، كيف أحل المشكلة).
# - "general": إذا كان السؤال عبارة عن تحية، سؤال عام لا يتطلب بحث (مثل "من أنت؟"، "كيف حالك؟")، حديث صغير، أو إهانة.

# أجب بصيغة JSON فقط، مع مفتاح "category".

# أمثلة:
# - سؤال المستخدم: "اشرح لي خطوات تثبيت البرنامج." -> {{"category": "technical"}}
# - سؤال المستخدم: "من هو مهدي أبو علي؟" -> {{"category": "technical"}}
# - سؤال المستخدم: "مرحباً يا ساعد" -> {{"category": "general"}}
# - سؤال المستخدم: "من تكون؟" -> {{"category": "general"}}

# سؤال المستخدم:
# {question}
# """

# # --- 2. قالب نظام RAG التقني ---
# RAG_PROMPT_TEMPLATE = """
# **مهمتك:** أنت مساعد دعم فني خبير ومختص لـ **{tenant_name}**. استخدم "السياق" التالي للإجابة على "سؤال المستخدم" بدقة.
# - إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال."
# - أجب دائمًا باللغة العربية.

# **السياق:**
# {context}

# **سؤال المستخدم:**
# {question}

# **الإجابة:**
# """

# # --- 3. قالب المحادثة العامة (مع شخصية ديناميكية) ---
# GENERAL_PROMPT_TEMPLATE = """
# **مهمتك:** أنت "ساعد"، المساعد الآلي لـ **{tenant_name}**. أنت ذكي وودود. تفاعل مع "سؤال المستخدم" بطريقة مناسبة ومهذبة.
# - إذا كان السؤال "من أنت؟" أو ما شابه: عرّف بنفسك: "أنا ساعد، مساعد الدعم الآلي لـ {tenant_name}. كيف يمكنني خدمتك؟"
# - إذا كان السؤال تحية: رد التحية بلطف. (مثال: "وعليكم السلام! أهلاً بك في خدمة الدعم لـ {tenant_name}.")
# - إذا كان السؤال إهانة: حافظ على هدوئك ورد باحترافية: "أنا هنا لمساعدتك في أي استفسارات لديك حول {tenant_name}."
# - أجب دائمًا باللغة العربية.

# سؤال المستخدم:
# {question}
# """

# # -----------------------------------------------------------------------------
# # 🌍 المتغيرات العالمية وسلاسل العمل
# # -----------------------------------------------------------------------------
# vector_store: FAISS = None
# llm: Ollama = None
# embeddings_model: OllamaEmbeddings = None
# all_docs_for_bm25: List[Document] = []
# cross_encoder: CrossEncoder = None
# full_rag_chain = None
# general_chain = None
# routing_chain = None

# # -----------------------------------------------------------------------------
# # 🚀 تهيئة الوكيل (مع إعادة التوجيه)
# # -----------------------------------------------------------------------------
# def initialize_agent():
#     global vector_store, llm, embeddings_model, all_docs_for_bm25, cross_encoder, full_rag_chain, general_chain, routing_chain
#     if routing_chain:
#         logging.info("✅ الوكيل الذكي (مع التوجيه) مُهيأ مسبقًا.")
#         return
    
#     try:
#         logging.info("=" * 80)
#         logging.info("🚀 بدء تهيئة الوكيل الذكي (مع التوجيه والشخصية الديناميكية)...")
        
#         llm = Ollama(model=CHAT_MODEL_NAME, temperature=0.1)
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#         vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings_model, allow_dangerous_deserialization=True)
#         docstore_ids = list(vector_store.docstore._dict.keys())
#         all_docs_for_bm25 = [vector_store.docstore._dict[i] for i in docstore_ids]
#         cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        
#         # --- بناء السلاسل ---
#         rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
#         full_rag_chain = (
#             RunnablePassthrough.assign(context=lambda x: format_docs_with_source(x["docs"]))
#             | rag_prompt
#             | llm
#             | StrOutputParser()
#         )

#         general_prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
#         general_chain = general_prompt | llm | StrOutputParser()

#         routing_prompt = PromptTemplate.from_template(ROUTING_PROMPT_TEMPLATE)
#         routing_chain = routing_prompt | llm | JsonOutputParser()

#         logging.info(" اكتملت تهيئة الوكيل الذكي بنجاح! ✨")
#     except Exception as e:
#         logging.critical(f" فشل حاسم أثناء التهيئة: {e}", exc_info=True)
#         raise

# # -----------------------------------------------------------------------------
# # 헬 دوال مساعدة
# # -----------------------------------------------------------------------------
# def format_docs_with_source(docs: List[Document]) -> str:
#     """تنسق المستندات المسترجعة وتضيف المصادر."""
#     if not docs:
#         return "لا يوجد سياق متوفر."
#     sources = {doc.metadata.get("source", "مصدر غير معروف") for doc in docs}
#     formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
#     return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

# def perform_hybrid_retrieval_and_rerank(question: str, tenant_id: str, k: int) -> List[Document]:
#     """ينفذ البحث الهجين الكامل مع إعادة الترتيب."""
#     faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15, "filter": {"tenant_id": tenant_id}})
#     faiss_docs = faiss_retriever.invoke(question)
    
#     tenant_docs_indices = [i for i, doc in enumerate(all_docs_for_bm25) if doc.metadata.get("tenant_id") == tenant_id]
#     bm25_docs = []
#     if tenant_docs_indices:
#         tenant_corpus = [all_docs_for_bm25[i].page_content.split(" ") for i in tenant_docs_indices]
#         bm25_for_tenant = BM25Okapi(tenant_corpus)
#         tokenized_query = question.split(" ")
#         doc_scores = bm25_for_tenant.get_scores(tokenized_query)
#         top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:15]
#         bm25_docs = [all_docs_for_bm25[tenant_docs_indices[i]] for i in top_n_indices]
    
#     combined_docs_list = list({doc.page_content: doc for doc in faiss_docs + bm25_docs}.values())
#     if not combined_docs_list:
#         return []

#     model_input_pairs = [[question, doc.page_content] for doc in combined_docs_list]
#     scores = cross_encoder.predict(model_input_pairs)
#     docs_with_scores = sorted(zip(combined_docs_list, scores), key=lambda x: x[1], reverse=True)
    
#     return [doc for doc, score in docs_with_scores[:k]]

# # -----------------------------------------------------------------------------
# # 🧠 بث الإجابة (النسخة النهائية مع الهوية الديناميكية المستنبطة)
# # -----------------------------------------------------------------------------
# async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
#     if not routing_chain:
#         raise RuntimeError("⚠️ الوكيل الذكي غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")
    
#     logging.info(f"📩 استقبال سؤال من '{tenant_id}': {question}")
#     try:
#         # 1. مرحلة التوجيه
#         perf_logger.start("routing")
#         route_decision = await routing_chain.ainvoke({"question": question})
#         category = route_decision.get("category", "technical")
#         perf_logger.end("routing", tenant_id, question, extra_info={"decision": category})
#         logging.info(f"🧠 قرار التوجيه: '{category}'")

#         # 2. تنفيذ المسار
#         if category == "technical":
#             logging.info("🚀 تنفيذ مسار الدعم الفني (RAG)...")
#             perf_logger.start("retrieval_rerank")
#             final_docs = perform_hybrid_retrieval_and_rerank(question, tenant_id, k_results)
#             perf_logger.end("retrieval_rerank", tenant_id, question, extra_info={"final_doc_count": len(final_docs)})
            
#             # استنباط الهوية الديناميكية من المستندات المسترجعة
#             entity_name = "الخدمة" # اسم افتراضي
#             if final_docs and "entity_name" in final_docs[0].metadata:
#                 entity_name = final_docs[0].metadata["entity_name"]
#             logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")
            
#             async for chunk in full_rag_chain.astream({"question": question, "docs": final_docs, "tenant_name": entity_name}):
#                 yield chunk
#         else: # general
#             logging.info("💬 تنفيذ مسار المحادثة العامة...")
            
#             # استنباط الهوية الديناميكية عبر بحث خفيف جداً
#             temp_docs = vector_store.similarity_search("", filter={"tenant_id": tenant_id}, k=1)
#             entity_name = "الخدمة" # اسم افتراضي
#             if temp_docs and "entity_name" in temp_docs[0].metadata:
#                 entity_name = temp_docs[0].metadata["entity_name"]
#             logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")

#             async for chunk in general_chain.astream({"question": question, "tenant_name": entity_name}):
#                 yield chunk
#     except Exception as e:
#         logging.error(f"❌ خطأ أثناء بث الإجابة: {e}", exc_info=True)
#         yield "عذرًا، حدث خطأ داخلي أثناء معالجة سؤالك."
#         perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})


# core_logic.py (النسخة النهائية فائقة السرعة)
#

# # #
# import os
# import logging
# from typing import List, AsyncGenerator
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import langchain
# from langchain_core.caches import InMemoryCache
# from langchain_core.documents import Document
# from sentence_transformers import CrossEncoder
# from rank_bm25 import BM25Okapi
# # 🔴🔴🔴 --- استيراد جديد ومهم --- 🔴🔴🔴
# from transformers import pipeline

# from .performance_tracker import PerformanceLogger

# # -----------------------------------------------------------------------------
# # 🧩 إعدادات عامة وتسجيل
# # -----------------------------------------------------------------------------
# perf_logger = PerformanceLogger()
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))
# langchain.llm_cache = InMemoryCache()

# # -----------------------------------------------------------------------------
# # 📦 متغيرات البيئة والنماذج
# # -----------------------------------------------------------------------------
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
# VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))
# RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
# # 🔴🔴🔴 --- اسم نموذج التصنيف السريع --- 🔴🔴🔴
# CLASSIFIER_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# # -----------------------------------------------------------------------------
# # 🧠 قوالب الـ Prompts (لم نعد نحتاج قالب التوجيه)
# # -----------------------------------------------------------------------------
# RAG_PROMPT_TEMPLATE = """
# **مهمتك:** أنت مساعد دعم فني خبير ومختص لـ **{tenant_name}**. استخدم "السياق" التالي للإجابة على "سؤال المستخدم" بدقة.
# - إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال."
# - أجب دائمًا باللغة العربية.
# **السياق:** {context}
# **سؤال المستخدم:** {question}
# **الإجابة:**"""

# GENERAL_PROMPT_TEMPLATE = """
# **مهمتك:** أنت "ساعد"، المساعد الآلي لـ **{tenant_name}**. أنت ذكي وودود. تفاعل مع "سؤال المستخدم" بطريقة مناسبة ومهذبة.
# - إذا كان السؤال "من أنت؟" أو ما شابه: عرّف بنفسك: "أنا ساعد، مساعد الدعم الآلي لـ {tenant_name}. كيف يمكنني خدمتك؟"
# - إذا كان السؤال تحية: رد التحية بلطف. (مثال: "وعليكم السلام! أهلاً بك في خدمة الدعم لـ {tenant_name}.")
# - إذا كان السؤال إهانة: حافظ على هدوئك ورد باحترافية: "أنا هنا لمساعدتك في أي استفسارات لديك حول {tenant_name}."
# - أجب دائمًا باللغة العربية.
# **سؤال المستخدم:** {question}
# """

# # -----------------------------------------------------------------------------
# # 🌍 المتغيرات العالمية
# # -----------------------------------------------------------------------------
# vector_store: FAISS = None
# llm: Ollama = None
# embeddings_model: OllamaEmbeddings = None
# all_docs_for_bm25: List[Document] = []
# cross_encoder: CrossEncoder = None
# full_rag_chain = None
# general_chain = None
# # 🔴🔴🔴 --- تم استبدال routing_chain بـ classifier --- 🔴🔴🔴
# classifier = None

# # -----------------------------------------------------------------------------
# # 🚀 تهيئة الوكيل (مع المصنف السريع)
# # -----------------------------------------------------------------------------
# def initialize_agent():
#     global vector_store, llm, embeddings_model, all_docs_for_bm25, cross_encoder, full_rag_chain, general_chain, classifier
#     if classifier:
#         logging.info("✅ الوكيل فائق السرعة مُهيأ مسبقًا.")
#         return
    
#     try:
#         logging.info("=" * 80)
#         logging.info("🚀 بدء تهيئة الوكيل فائق السرعة (مع مصنف مخصص)...")
        
#         # تحميل المكونات الأساسية
#         llm = Ollama(model=CHAT_MODEL_NAME, temperature=0.1)
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#         vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings_model, allow_dangerous_deserialization=True)
#         docstore_ids = list(vector_store.docstore._dict.keys())
#         all_docs_for_bm25 = [vector_store.docstore._dict[i] for i in docstore_ids]
#         cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        
#         # 🔴🔴🔴 --- تهيئة المصنف السريع --- 🔴🔴🔴
#         logging.info(f"[*] جارٍ تحميل نموذج التصنيف السريع: '{CLASSIFIER_MODEL_NAME}'...")
#         classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL_NAME)
#         logging.info("[*] تم تحميل المصنف بنجاح.")

#         # بناء السلاسل
#         rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
#         full_rag_chain = (
#             RunnablePassthrough.assign(context=lambda x: format_docs_with_source(x["docs"]))
#             | rag_prompt
#             | llm
#             | StrOutputParser()
#         )

#         general_prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
#         general_chain = general_prompt | llm | StrOutputParser()

#         logging.info("✨ اكتملت تهيئة الوكيل فائق السرعة بنجاح! ✨")
#     except Exception as e:
#         logging.critical(f"❌ فشل حاسم أثناء التهيئة: {e}", exc_info=True)
#         raise

# # -----------------------------------------------------------------------------
# # 헬 دوال مساعدة (بدون تغيير)
# # -----------------------------------------------------------------------------
# def format_docs_with_source(docs: List[Document]) -> str:
#     # ... (نفس الكود)
#     if not docs: return "لا يوجد سياق متوفر."
#     sources = {doc.metadata.get("source", "مصدر غير معروف") for doc in docs}
#     formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
#     return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

# def perform_hybrid_retrieval_and_rerank(question: str, tenant_id: str, k: int) -> List[Document]:
#     # ... (نفس الكود)
#     faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15, "filter": {"tenant_id": tenant_id}})
#     faiss_docs = faiss_retriever.invoke(question)
#     tenant_docs_indices = [i for i, doc in enumerate(all_docs_for_bm25) if doc.metadata.get("tenant_id") == tenant_id]
#     bm25_docs = []
#     if tenant_docs_indices:
#         tenant_corpus = [all_docs_for_bm25[i].page_content.split(" ") for i in tenant_docs_indices]
#         bm25_for_tenant = BM25Okapi(tenant_corpus)
#         tokenized_query = question.split(" ")
#         doc_scores = bm25_for_tenant.get_scores(tokenized_query)
#         top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:15]
#         bm25_docs = [all_docs_for_bm25[tenant_docs_indices[i]] for i in top_n_indices]
#     combined_docs_list = list({doc.page_content: doc for doc in faiss_docs + bm25_docs}.values())
#     if not combined_docs_list: return []
#     model_input_pairs = [[question, doc.page_content] for doc in combined_docs_list]
#     scores = cross_encoder.predict(model_input_pairs)
#     docs_with_scores = sorted(zip(combined_docs_list, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, score in docs_with_scores[:k]]

# # -----------------------------------------------------------------------------
# # 🧠 بث الإجابة (النسخة فائقة السرعة)
# # -----------------------------------------------------------------------------
# async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
#     if not classifier:
#         raise RuntimeError("⚠️ الوكيل فائق السرعة غير مُهيأ.")
    
#     logging.info(f"📩 استقبال سؤال من '{tenant_id}': {question}")
#     try:
#         # 🔴🔴🔴 --- 1. مرحلة التوجيه فائقة السرعة --- 🔴🔴🔴
#         perf_logger.start("routing")
#         candidate_labels = ["سؤال تقني", "محادثة عامة"]
#         # ملاحظة: لا نستخدم ainvoke هنا لأن pipeline لا تدعمها افتراضيًا
#         result = classifier(question, candidate_labels, multi_label=False)
#         # أعلى تصنيف هو القرار
#         decision = result['labels'][0]
#         category = "technical" if decision == "سؤال تقني" else "general"
#         perf_logger.end("routing", tenant_id, question, extra_info={"decision": category, "score": result['scores'][0]})
#         logging.info(f"🧠 قرار التوجيه فائق السرعة: '{category}' (بثقة: {result['scores'][0]:.2f})")

#         # 2. تنفيذ المسار (نفس المنطق السابق)
#         if category == "technical":
#             logging.info("🚀 تنفيذ مسار الدعم الفني (RAG)...")
#             perf_logger.start("retrieval_rerank")
#             final_docs = perform_hybrid_retrieval_and_rerank(question, tenant_id, k_results)
#             perf_logger.end("retrieval_rerank", tenant_id, question, extra_info={"final_doc_count": len(final_docs)})
            
#             entity_name = "الخدمة"
#             if final_docs and "entity_name" in final_docs[0].metadata:
#                 entity_name = final_docs[0].metadata["entity_name"]
#             logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")
            
#             async for chunk in full_rag_chain.astream({"question": question, "docs": final_docs, "tenant_name": entity_name}):
#                 yield chunk
#         else: # general
#             logging.info("💬 تنفيذ مسار المحادثة العامة...")
#             temp_docs = vector_store.similarity_search("", filter={"tenant_id": tenant_id}, k=1)
#             entity_name = "الخدمة"
#             if temp_docs and "entity_name" in temp_docs[0].metadata:
#                 entity_name = temp_docs[0].metadata["entity_name"]
#             logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")

#             async for chunk in general_chain.astream({"question": question, "tenant_name": entity_name}):
#                 yield chunk
#     except Exception as e:
#         logging.error(f"❌ خطأ أثناء بث الإجابة: {e}", exc_info=True)
#         yield "عذرًا، حدث خطأ داخلي أثناء معالجة سؤالك."
#         perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})

# # core_logic.py (النسخة النهائية فائقة السرعة)

# import os
# import logging
# from typing import List, AsyncGenerator
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import langchain
# from langchain_core.caches import InMemoryCache
# from langchain_core.documents import Document
# from sentence_transformers import CrossEncoder
# from rank_bm25 import BM25Okapi
# # 🔴🔴🔴 --- استيراد جديد ومهم --- 🔴🔴🔴
# from transformers import pipeline

# from .performance_tracker import PerformanceLogger

# # -----------------------------------------------------------------------------
# # 🧩 إعدادات عامة وتسجيل
# # -----------------------------------------------------------------------------
# perf_logger = PerformanceLogger()
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))
# langchain.llm_cache = InMemoryCache()

# # -----------------------------------------------------------------------------
# # 📦 متغيرات البيئة والنماذج
# # -----------------------------------------------------------------------------
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
# VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))
# RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
# # 🔴🔴🔴 --- اسم نموذج التصنيف السريع --- 🔴🔴🔴
# CLASSIFIER_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# # -----------------------------------------------------------------------------
# # 🧠 قوالب الـ Prompts (لم نعد نحتاج قالب التوجيه)
# # -----------------------------------------------------------------------------
# RAG_PROMPT_TEMPLATE = """
# **مهمتك:** أنت مساعد دعم فني خبير ومختص لـ **{tenant_name}**. استخدم "السياق" التالي للإجابة على "سؤال المستخدم" بدقة.
# - إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال."
# - أجب دائمًا باللغة العربية.
# **السياق:** {context}
# **سؤال المستخدم:** {question}
# **الإجابة:**"""

# GENERAL_PROMPT_TEMPLATE = """
# **مهمتك:** أنت "ساعد"، المساعد الآلي لـ **{tenant_name}**. أنت ذكي وودود. تفاعل مع "سؤال المستخدم" بطريقة مناسبة ومهذبة.
# - إذا كان السؤال "من أنت؟" أو ما شابه: عرّف بنفسك: "أنا ساعد، مساعد الدعم الآلي لـ {tenant_name}. كيف يمكنني خدمتك؟"
# - إذا كان السؤال تحية: رد التحية بلطف. (مثال: "وعليكم السلام! أهلاً بك في خدمة الدعم لـ {tenant_name}.")
# - إذا كان السؤال إهانة: حافظ على هدوئك ورد باحترافية: "أنا هنا لمساعدتك في أي استفسارات لديك حول {tenant_name}."
# - أجب دائمًا باللغة العربية.
# **سؤال المستخدم:** {question}
# """

# # -----------------------------------------------------------------------------
# # 🌍 المتغيرات العالمية
# # -----------------------------------------------------------------------------
# vector_store: FAISS = None
# llm: Ollama = None
# embeddings_model: OllamaEmbeddings = None
# all_docs_for_bm25: List[Document] = []
# cross_encoder: CrossEncoder = None
# full_rag_chain = None
# general_chain = None
# # 🔴🔴🔴 --- تم استبدال routing_chain بـ classifier --- 🔴🔴🔴
# classifier = None

# # -----------------------------------------------------------------------------
# # 🚀 تهيئة الوكيل (مع المصنف السريع)
# # -----------------------------------------------------------------------------
# def initialize_agent():
#     global vector_store, llm, embeddings_model, all_docs_for_bm25, cross_encoder, full_rag_chain, general_chain, classifier
#     if classifier:
#         logging.info("✅ الوكيل فائق السرعة مُهيأ مسبقًا.")
#         return
    
#     try:
#         logging.info("=" * 80)
#         logging.info("🚀 بدء تهيئة الوكيل فائق السرعة (مع مصنف مخصص)...")
        
#         # تحميل المكونات الأساسية
#         llm = Ollama(model=CHAT_MODEL_NAME, temperature=0.1)
#         embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#         vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings_model, allow_dangerous_deserialization=True)
#         docstore_ids = list(vector_store.docstore._dict.keys())
#         all_docs_for_bm25 = [vector_store.docstore._dict[i] for i in docstore_ids]
#         cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        
#         # 🔴🔴🔴 --- تهيئة المصنف السريع --- 🔴🔴🔴
#         logging.info(f"[*] جارٍ تحميل نموذج التصنيف السريع: '{CLASSIFIER_MODEL_NAME}'...")
#         classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL_NAME)
#         logging.info("[*] تم تحميل المصنف بنجاح.")

#         # بناء السلاسل
#         rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
#         full_rag_chain = (
#             RunnablePassthrough.assign(context=lambda x: format_docs_with_source(x["docs"]))
#             | rag_prompt
#             | llm
#             | StrOutputParser()
#         )

#         general_prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
#         general_chain = general_prompt | llm | StrOutputParser()

#         logging.info("✨ اكتملت تهيئة الوكيل فائق السرعة بنجاح! ✨")
#     except Exception as e:
#         logging.critical(f"❌ فشل حاسم أثناء التهيئة: {e}", exc_info=True)
#         raise

# # -----------------------------------------------------------------------------
# # 헬 دوال مساعدة (بدون تغيير)
# # -----------------------------------------------------------------------------
# def format_docs_with_source(docs: List[Document]) -> str:
#     # ... (نفس الكود)
#     if not docs: return "لا يوجد سياق متوفر."
#     sources = {doc.metadata.get("source", "مصدر غير معروف") for doc in docs}
#     formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
#     return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

# def perform_hybrid_retrieval_and_rerank(question: str, tenant_id: str, k: int) -> List[Document]:
#     # ... (نفس الكود)
#     faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15, "filter": {"tenant_id": tenant_id}})
#     faiss_docs = faiss_retriever.invoke(question)
#     tenant_docs_indices = [i for i, doc in enumerate(all_docs_for_bm25) if doc.metadata.get("tenant_id") == tenant_id]
#     bm25_docs = []
#     if tenant_docs_indices:
#         tenant_corpus = [all_docs_for_bm25[i].page_content.split(" ") for i in tenant_docs_indices]
#         bm25_for_tenant = BM25Okapi(tenant_corpus)
#         tokenized_query = question.split(" ")
#         doc_scores = bm25_for_tenant.get_scores(tokenized_query)
#         top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:15]
#         bm25_docs = [all_docs_for_bm25[tenant_docs_indices[i]] for i in top_n_indices]
#     combined_docs_list = list({doc.page_content: doc for doc in faiss_docs + bm25_docs}.values())
#     if not combined_docs_list: return []
#     model_input_pairs = [[question, doc.page_content] for doc in combined_docs_list]
#     scores = cross_encoder.predict(model_input_pairs)
#     docs_with_scores = sorted(zip(combined_docs_list, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, score in docs_with_scores[:k]]

# # -----------------------------------------------------------------------------
# # 🧠 بث الإجابة (النسخة فائقة السرعة)
# # -----------------------------------------------------------------------------
# async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
#     if not classifier:
#         raise RuntimeError("⚠️ الوكيل فائق السرعة غير مُهيأ.")
    
#     logging.info(f"📩 استقبال سؤال من '{tenant_id}': {question}")
#     try:
#         # 🔴🔴🔴 --- 1. مرحلة التوجيه فائقة السرعة --- 🔴🔴🔴
#         perf_logger.start("routing")
#         candidate_labels = ["سؤال تقني", "محادثة عامة"]
#         # ملاحظة: لا نستخدم ainvoke هنا لأن pipeline لا تدعمها افتراضيًا
#         result = classifier(question, candidate_labels, multi_label=False)
#         # أعلى تصنيف هو القرار
#         decision = result['labels'][0]
#         category = "technical" if decision == "سؤال تقني" else "general"
#         perf_logger.end("routing", tenant_id, question, extra_info={"decision": category, "score": result['scores'][0]})
#         logging.info(f"🧠 قرار التوجيه فائق السرعة: '{category}' (بثقة: {result['scores'][0]:.2f})")

#         # 2. تنفيذ المسار (نفس المنطق السابق)
#         if category == "technical":
#             logging.info("🚀 تنفيذ مسار الدعم الفني (RAG)...")
#             perf_logger.start("retrieval_rerank")
#             final_docs = perform_hybrid_retrieval_and_rerank(question, tenant_id, k_results)
#             perf_logger.end("retrieval_rerank", tenant_id, question, extra_info={"final_doc_count": len(final_docs)})
            
#             entity_name = "الخدمة"
#             if final_docs and "entity_name" in final_docs[0].metadata:
#                 entity_name = final_docs[0].metadata["entity_name"]
#             logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")
            
#             async for chunk in full_rag_chain.astream({"question": question, "docs": final_docs, "tenant_name": entity_name}):
#                 yield chunk
#         else: # general
#             logging.info("💬 تنفيذ مسار المحادثة العامة...")
#             temp_docs = vector_store.similarity_search("", filter={"tenant_id": tenant_id}, k=1)
#             entity_name = "الخدمة"
#             if temp_docs and "entity_name" in temp_docs[0].metadata:
#                 entity_name = temp_docs[0].metadata["entity_name"]
#             logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")

#             async for chunk in general_chain.astream({"question": question, "tenant_name": entity_name}):
#                 yield chunk
#     except Exception as e:
#         logging.error(f"❌ خطأ أثناء بث الإجابة: {e}", exc_info=True)
#         yield "عذرًا، حدث خطأ داخلي أثناء معالجة سؤالك."
#         perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})

# /2_central_api_service/agent_app/core_logic.py (النسخة النهائية مع تحسين المصنف)

# /2_central_api_service/agent_app/core_logic.py (النسخة النهائية مع هوية الدعم الفني المتخصص)
# /2_central_api_service/agent_app/core_logic.py (النسخة النهائية الكاملة)

import os
import logging
from typing import List, AsyncGenerator, Dict, Any, Literal

# --- استيراد مكتبات LangChain والمجتمع ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.caches import InMemoryCache
import langchain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- استيراد مكتبات البحث وإعادة الترتيب ---
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline

# --- استيراد الوحدات المحلية ---
from .performance_tracker import PerformanceLogger

# =================================================================================
# 1. الإعدادات الأولية والأساسية (Configuration & Setup)
# =================================================================================

# --- إعداد نظام التسجيل (Logging) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
)

# --- تفعيل الذاكرة المؤقتة لتحسين الأداء ---
langchain.llm_cache = InMemoryCache()
logging.info("تم تفعيل الذاكرة المؤقتة (InMemoryCache) لـ LangChain.")

# --- تحميل متغيرات البيئة ---
from dotenv import load_dotenv
load_dotenv()

# --- تعريف الثوابت ونماذج الذكاء الاصطناعي ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "default_embedding_model")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "default_chat_model")
RERANK_MODEL = "BAAI/bge-reranker-base"
CLASSIFIER_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))

# --- إعداد مسجل الأداء ---
perf_logger = PerformanceLogger()

# =================================================================================
# 2. قوالب التوجيه (Prompts) المحسّنة لهوية الدعم الفني
# =================================================================================

# --- قالب الإجابة الفنية (RAG) ---
RAG_PROMPT_TEMPLATE = """
### المهمة الأساسية ###
أنت "ساعد"، مساعد الدعم الفني الذكي والمتخصص في نظام **{tenant_name}**. مهمتك هي تحليل سؤال المستخدم وتقديم حلول وإجابات دقيقة بالاعتماد **فقط** على المعلومات التقنية المتوفرة في قسم "السياق".

### قواعد صارمة ###
1.  **الالتزام بالسياق:** لا تستخدم أي معلومات خارج قاعدة المعرفة التقنية المتاحة في السياق.
2.  **حل المشاكل:** ركز على تقديم خطوات عملية، إرشادات، أو تفسيرات تقنية تساعد المستخدم على حل مشكلته أو فهم النظام.
3.  **عدم وجود معلومات:** إذا كانت الإجابة غير موجودة، أجب حصريًا: "عفواً، لا أملك المعلومات الكافية حول هذه الجزئية في نظام {tenant_name}. هل يمكنك إعادة صياغة السؤال، أو هل تود توجيهك لخيارات دعم متقدمة؟"
4.  **اللغة:** أجب دائمًا بلغة عربية واضحة وموجهة للمستخدم التقني.

### السياق (قاعدة المعرفة التقنية للنظام) ###
{context}

### سؤال المستخدم ###
{question}

### الإجابة الفنية ###
"""

# --- قالب المحادثة العامة وتعريف الهوية ---
GENERAL_PROMPT_TEMPLATE = """
### المهمة الأساسية ###
أنت "ساعد"، مساعد الدعم الفني الآلي لنظام **{tenant_name}**. مهمتك هي التفاعل باحترافية وتوجيه المستخدم نحو طرح استفساراته الفنية.

### قواعد التفاعل ###
- **التعريف بالهوية:** إذا سُئلت "من أنت؟" أو ما شابه، أجب: "أنا ساعد، مساعد الدعم الفني الذكي لنظام {tenant_name}. أنا هنا لمساعدتك في حل المشاكل والإجابة على استفساراتك التقنية المتعلقة بالنظام."
- **التحية:** رد على التحيات بشكل احترافي ومباشر، مثل: "أهلاً بك في خدمة الدعم الفني لنظام {tenant_name}. كيف يمكنني مساعدتك اليوم؟"
- **الأسئلة خارج النطاق:** إذا كان السؤال عامًا جدًا ولا يتعلق بالدعم الفني، وجه المستخدم بلطف: "مهمتي الأساسية هي تقديم الدعم الفني لنظام {tenant_name}. هل لديك استفسار تقني أو مشكلة تواجهك داخل النظام؟"
- **التعامل مع الإساءة أو الكلام غير المفهوم:** إذا كان الإدخال عبارة عن إهانة أو كلام غير مترابط، أجب باحترافية وهدوء: "أنا هنا لتقديم المساعدة الفنية. يرجى طرح استفسارك بوضوح حتى أتمكن من مساعدتك."
- **اللغة:** استخدم اللغة العربية الرسمية دائمًا.

### سؤال المستخدم ###
{question}

### الإجابة ###
"""

# --- قالب توجيه المستخدم عند الحاجة للمساعدة ---
FALLBACK_PROMPT_TEMPLATE = """
عفواً، لم أتمكن من العثور على إجابة دقيقة في قاعدة المعرفة.

**خيارات المساعدة:**
1.  **إعادة صياغة السؤال:** قد يساعد استخدام كلمات مختلفة في العثور على الإجابة.
2.  **زيارة مركز المساعدة:** يمكنك تصفح التوثيقات الكاملة عبر الرابط التالي: [أدخل رابط التوثيقات هنا]
3.  **التواصل مع الدعم الفني:** إذا استمرت المشكلة، يمكنك التواصل مباشرة مع فريق الدعم البشري.

هل تود تجربة خيار آخر؟
"""

# =================================================================================
# 3. المتغيرات العالمية وسلاسل العمل (Global State & Chains)
# =================================================================================

vector_store: FAISS | None = None
llm: Ollama | None = None
embeddings_model: OllamaEmbeddings | None = None
cross_encoder: CrossEncoder | None = None
classifier: Any | None = None
all_docs_for_bm25: List[Document] = []
rag_chain: Any = None
general_chain: Any = None
fallback_chain: Any = None

# =================================================================================
# 4. دالة التهيئة الشاملة (Initialization Function)
# =================================================================================

def initialize_agent():
    """
    تقوم بتهيئة جميع مكونات الوكيل (النماذج، قواعد البيانات، السلاسل) مرة واحدة عند بدء التشغيل.
    """
    global vector_store, llm, embeddings_model, cross_encoder, classifier, all_docs_for_bm25
    global rag_chain, general_chain, fallback_chain

    if rag_chain:
        logging.info("الوكيل مُهيأ بالفعل وجاهز للعمل.")
        return

    logging.info("بدء تهيئة وكيل الدعم الفني الذكي...")

    try:
        logging.info(f"تحميل نموذج المحادثة: {CHAT_MODEL}")
        llm = Ollama(model=CHAT_MODEL, temperature=0.1)
        
        logging.info(f"تحميل نموذج التضمين: {EMBEDDING_MODEL}")
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

        logging.info(f"تحميل نموذج إعادة الترتيب: {RERANK_MODEL}")
        cross_encoder = CrossEncoder(RERANK_MODEL)

        logging.info(f"تحميل مصنف الأسئلة: {CLASSIFIER_MODEL}")
        classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)

        if not os.path.exists(VECTOR_DB_PATH):
            logging.error(f"خطأ فادح: مجلد قاعدة المعرفة غير موجود في المسار: {VECTOR_DB_PATH}")
            raise FileNotFoundError("مجلد قاعدة المعرفة مفقود.")
        
        logging.info(f"تحميل قاعدة المعرفة من: {VECTOR_DB_PATH}")
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        all_docs_for_bm25 = list(vector_store.docstore._dict.values())
        logging.info(f"تم تحميل قاعدة المعرفة بنجاح ({len(all_docs_for_bm25)} مستند).")

        logging.info("بناء سلاسل العمل المنطقية...")
        
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["docs"]))
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        general_prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
        general_chain = general_prompt | llm | StrOutputParser()

        fallback_prompt = PromptTemplate.from_template(FALLBACK_PROMPT_TEMPLATE)
        fallback_chain = fallback_prompt | llm | StrOutputParser()

        logging.info("اكتملت تهيئة الوكيل بنجاح وهو الآن جاهز لاستقبال الطلبات.")

    except Exception as e:
        logging.critical(f"فشل حاسم أثناء تهيئة الوكيل: {e}", exc_info=True)
        raise

# =================================================================================
# 5. الدوال المساعدة والمنطق الداخلي (Helper & Logic Functions)
# =================================================================================

def _format_docs(docs: List[Document]) -> str:
    """تنسق المستندات المسترجعة لتقديمها كـ "سياق" للنموذج اللغوي."""
    if not docs:
        return "لا توجد معلومات متاحة."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def _get_dynamic_identity(tenant_id: str) -> str:
    """تستنبط اسم النظام (الهوية الديناميكية) من قاعدة المعرفة."""
    if not vector_store: return "النظام الحالي"
    docs = vector_store.similarity_search("", filter={"tenant_id": tenant_id}, k=1)
    if docs and "entity_name" in docs[0].metadata:
        return docs[0].metadata["entity_name"]
    return "النظام الحالي"

def _classify_question(question: str) -> Literal["technical", "general", "inappropriate"]:
    """
    يستخدم مصنفًا سريعًا لتحديد نية المستخدم إلى ثلاث فئات.
    """
    if not classifier: raise RuntimeError("المصنف غير مهيأ.")
    
    perf_logger.start("routing")
    
    labels = [
        "سؤال فني أو استفسار عن معلومات محددة", 
        "تحية، شكر، أو سؤال عام عن الهوية مثل من أنت",
        "إهانة، كلام بذيء، أو عبارات عشوائية غير مفهومة"
    ]
    
    result = classifier(question, labels, multi_label=False)
    
    top_label = result['labels'][0]
    decision: Literal["technical", "general", "inappropriate"]
    if top_label == labels[0]:
        decision = "technical"
    elif top_label == labels[1]:
        decision = "general"
    else:
        decision = "inappropriate"
    
    perf_logger.end("routing", "N/A", question, {"decision": decision, "confidence": result['scores'][0]})
    logging.info(f"قرار التوجيه: '{decision}' (بثقة: {result['scores'][0]:.2f})")
    
    return decision

def _hybrid_retrieval_and_rerank(question: str, tenant_id: str, k: int) -> List[Document]:
    """تنفذ استراتيجية بحث هجينة ثم تعيد ترتيب النتائج."""
    if not vector_store or not cross_encoder: raise RuntimeError("مكونات البحث غير مهيأة.")
    
    perf_logger.start("retrieval_rerank")
    
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k * 5, 'filter': {'tenant_id': tenant_id}}
    )
    faiss_docs = faiss_retriever.invoke(question)

    tenant_docs = [doc for doc in all_docs_for_bm25 if doc.metadata.get("tenant_id") == tenant_id]
    bm25_docs = []
    if tenant_docs:
        corpus = [doc.page_content.split() for doc in tenant_docs]
        bm25 = BM25Okapi(corpus)
        tokenized_query = question.split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k * 5]
        bm25_docs = [tenant_docs[i] for i in top_indices]

    combined_docs = list({doc.page_content: doc for doc in faiss_docs + bm25_docs}.values())
    if not combined_docs:
        perf_logger.end("retrieval_rerank", tenant_id, question, {"status": "no_docs_found"})
        return []

    pairs = [[question, doc.page_content] for doc in combined_docs]
    scores = cross_encoder.predict(pairs)
    
    reranked_results = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True)
    
    final_docs = [doc for score, doc in reranked_results[:k]]
    
    perf_logger.end("retrieval_rerank", tenant_id, question, {"retrieved_count": len(final_docs)})
    logging.info(f"تم استرجاع وإعادة ترتيب {len(final_docs)} مستندًا ذا صلة.")
    
    return final_docs

# =================================================================================
# 6. نقطة الدخول الرئيسية (Main Entrypoint)
# =================================================================================

async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
    """
    الدالة الرئيسية التي تعالج سؤال المستخدم وتبث الإجابة بشكل تفاعلي.
    """
    if not rag_chain or not general_chain or not fallback_chain:
        raise RuntimeError("الوكيل غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")

    logging.info(f"استلام طلب جديد من العميل '{tenant_id}'.")
    
    try:
        category = _classify_question(question)
        tenant_name = _get_dynamic_identity(tenant_id)
        logging.info(f"الهوية الديناميكية المحددة: '{tenant_name}'")

        if category == "technical":
            logging.info("تنفيذ مسار الدعم الفني (RAG)...")
            relevant_docs = _hybrid_retrieval_and_rerank(question, tenant_id, k_results)
            
            if not relevant_docs:
                logging.warning("لم يتم العثور على مستندات ذات صلة. سيتم استخدام إجابة الطوارئ.")
                async for chunk in fallback_chain.astream({}):
                    yield chunk
                return

            async for chunk in rag_chain.astream({
                "question": question,
                "docs": relevant_docs,
                "tenant_name": tenant_name
            }):
                yield chunk
        
        elif category == "inappropriate":
            logging.info("تنفيذ مسار الرد على المدخلات غير الملائمة...")
            async for chunk in general_chain.astream({
                "question": question,
                "tenant_name": tenant_name
            }):
                yield chunk

        else: # category == "general"
            logging.info("تنفيذ مسار المحادثة العامة...")
            async for chunk in general_chain.astream({
                "question": question,
                "tenant_name": tenant_name
            }):
                yield chunk

    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}", exc_info=True)
        yield "عذرًا، حدث خطأ فني. فريقنا يعمل على إصلاحه."
        perf_logger.end("error", tenant_id, question, {"error": str(e)})
# /2_central_api_service/agent_app/core_logic.py (النسخة النهائية الكاملة)

import os
import logging
from typing import List, AsyncGenerator, Dict, Any, Literal

# --- استيراد مكتبات LangChain والمجتمع ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.caches import InMemoryCache
import langchain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- استيراد مكتبات البحث وإعادة الترتيب ---
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline

# --- استيراد الوحدات المحلية ---
from .performance_tracker import PerformanceLogger

# =================================================================================
# 1. الإعدادات الأولية والأساسية (Configuration & Setup)
# =================================================================================

# --- إعداد نظام التسجيل (Logging) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
)

# --- تفعيل الذاكرة المؤقتة لتحسين الأداء ---
langchain.llm_cache = InMemoryCache()
logging.info("تم تفعيل الذاكرة المؤقتة (InMemoryCache) لـ LangChain.")

# --- تحميل متغيرات البيئة ---
from dotenv import load_dotenv
load_dotenv()

# --- تعريف الثوابت ونماذج الذكاء الاصطناعي ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "default_embedding_model")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "default_chat_model")
RERANK_MODEL = "BAAI/bge-reranker-base"
CLASSIFIER_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))

# --- إعداد مسجل الأداء ---
perf_logger = PerformanceLogger()

# =================================================================================
# 2. قوالب التوجيه (Prompts) المحسّنة لهوية الدعم الفني
# =================================================================================

# --- قالب الإجابة الفنية (RAG) ---
RAG_PROMPT_TEMPLATE = """
### المهمة الأساسية ###
أنت "ساعد"، مساعد الدعم الفني الذكي والمتخصص في نظام **{tenant_name}**. مهمتك هي تحليل سؤال المستخدم وتقديم حلول وإجابات دقيقة بالاعتماد **فقط** على المعلومات التقنية المتوفرة في قسم "السياق".

### قواعد صارمة ###
1.  **الالتزام بالسياق:** لا تستخدم أي معلومات خارج قاعدة المعرفة التقنية المتاحة في السياق.
2.  **حل المشاكل:** ركز على تقديم خطوات عملية، إرشادات، أو تفسيرات تقنية تساعد المستخدم على حل مشكلته أو فهم النظام.
3.  **عدم وجود معلومات:** إذا كانت الإجابة غير موجودة، أجب حصريًا: "عفواً، لا أملك المعلومات الكافية حول هذه الجزئية في نظام {tenant_name}. هل يمكنك إعادة صياغة السؤال، أو هل تود توجيهك لخيارات دعم متقدمة؟"
4.  **اللغة:** أجب دائمًا بلغة عربية واضحة وموجهة للمستخدم التقني.

### السياق (قاعدة المعرفة التقنية للنظام) ###
{context}

### سؤال المستخدم ###
{question}

### الإجابة الفنية ###
"""

# --- قالب المحادثة العامة وتعريف الهوية ---
GENERAL_PROMPT_TEMPLATE = """
### المهمة الأساسية ###
أنت "ساعد"، مساعد الدعم الفني الآلي لنظام **{tenant_name}**. مهمتك هي التفاعل باحترافية وتوجيه المستخدم نحو طرح استفساراته الفنية.

### قواعد التفاعل ###
- **التعريف بالهوية:** إذا سُئلت "من أنت؟" أو ما شابه، أجب: "أنا ساعد، مساعد الدعم الفني الذكي لنظام {tenant_name}. أنا هنا لمساعدتك في حل المشاكل والإجابة على استفساراتك التقنية المتعلقة بالنظام."
- **التحية:** رد على التحيات بشكل احترافي ومباشر، مثل: "أهلاً بك في خدمة الدعم الفني لنظام {tenant_name}. كيف يمكنني مساعدتك اليوم؟"
- **الأسئلة خارج النطاق:** إذا كان السؤال عامًا جدًا ولا يتعلق بالدعم الفني، وجه المستخدم بلطف: "مهمتي الأساسية هي تقديم الدعم الفني لنظام {tenant_name}. هل لديك استفسار تقني أو مشكلة تواجهك داخل النظام؟"
- **التعامل مع الإساءة أو الكلام غير المفهوم:** إذا كان الإدخال عبارة عن إهانة أو كلام غير مترابط، أجب باحترافية وهدوء: "أنا هنا لتقديم المساعدة الفنية. يرجى طرح استفسارك بوضوح حتى أتمكن من مساعدتك."
- **اللغة:** استخدم اللغة العربية الرسمية دائمًا.

### سؤال المستخدم ###
{question}

### الإجابة ###
"""

# --- قالب توجيه المستخدم عند الحاجة للمساعدة ---
FALLBACK_PROMPT_TEMPLATE = """
عفواً، لم أتمكن من العثور على إجابة دقيقة في قاعدة المعرفة.

**خيارات المساعدة:**
1.  **إعادة صياغة السؤال:** قد يساعد استخدام كلمات مختلفة في العثور على الإجابة.
2.  **زيارة مركز المساعدة:** يمكنك تصفح التوثيقات الكاملة عبر الرابط التالي: [أدخل رابط التوثيقات هنا]
3.  **التواصل مع الدعم الفني:** إذا استمرت المشكلة، يمكنك التواصل مباشرة مع فريق الدعم البشري.

هل تود تجربة خيار آخر؟
"""

# =================================================================================
# 3. المتغيرات العالمية وسلاسل العمل (Global State & Chains)
# =================================================================================

vector_store: FAISS | None = None
llm: Ollama | None = None
embeddings_model: OllamaEmbeddings | None = None
cross_encoder: CrossEncoder | None = None
classifier: Any | None = None
all_docs_for_bm25: List[Document] = []
rag_chain: Any = None
general_chain: Any = None
fallback_chain: Any = None

# =================================================================================
# 4. دالة التهيئة الشاملة (Initialization Function)
# =================================================================================

def initialize_agent():
    """
    تقوم بتهيئة جميع مكونات الوكيل (النماذج، قواعد البيانات، السلاسل) مرة واحدة عند بدء التشغيل.
    """
    global vector_store, llm, embeddings_model, cross_encoder, classifier, all_docs_for_bm25
    global rag_chain, general_chain, fallback_chain

    if rag_chain:
        logging.info("الوكيل مُهيأ بالفعل وجاهز للعمل.")
        return

    logging.info("بدء تهيئة وكيل الدعم الفني الذكي...")

    try:
        logging.info(f"تحميل نموذج المحادثة: {CHAT_MODEL}")
        llm = Ollama(model=CHAT_MODEL, temperature=0.1)
        
        logging.info(f"تحميل نموذج التضمين: {EMBEDDING_MODEL}")
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

        logging.info(f"تحميل نموذج إعادة الترتيب: {RERANK_MODEL}")
        cross_encoder = CrossEncoder(RERANK_MODEL)

        logging.info(f"تحميل مصنف الأسئلة: {CLASSIFIER_MODEL}")
        classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)

        if not os.path.exists(VECTOR_DB_PATH):
            logging.error(f"خطأ فادح: مجلد قاعدة المعرفة غير موجود في المسار: {VECTOR_DB_PATH}")
            raise FileNotFoundError("مجلد قاعدة المعرفة مفقود.")
        
        logging.info(f"تحميل قاعدة المعرفة من: {VECTOR_DB_PATH}")
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        all_docs_for_bm25 = list(vector_store.docstore._dict.values())
        logging.info(f"تم تحميل قاعدة المعرفة بنجاح ({len(all_docs_for_bm25)} مستند).")

        logging.info("بناء سلاسل العمل المنطقية...")
        
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["docs"]))
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        general_prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
        general_chain = general_prompt | llm | StrOutputParser()

        fallback_prompt = PromptTemplate.from_template(FALLBACK_PROMPT_TEMPLATE)
        fallback_chain = fallback_prompt | llm | StrOutputParser()

        logging.info("اكتملت تهيئة الوكيل بنجاح وهو الآن جاهز لاستقبال الطلبات.")

    except Exception as e:
        logging.critical(f"فشل حاسم أثناء تهيئة الوكيل: {e}", exc_info=True)
        raise

# =================================================================================
# 5. الدوال المساعدة والمنطق الداخلي (Helper & Logic Functions)
# =================================================================================

def _format_docs(docs: List[Document]) -> str:
    """تنسق المستندات المسترجعة لتقديمها كـ "سياق" للنموذج اللغوي."""
    if not docs:
        return "لا توجد معلومات متاحة."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def _get_dynamic_identity(tenant_id: str) -> str:
    """تستنبط اسم النظام (الهوية الديناميكية) من قاعدة المعرفة."""
    if not vector_store: return "النظام الحالي"
    docs = vector_store.similarity_search("", filter={"tenant_id": tenant_id}, k=1)
    if docs and "entity_name" in docs[0].metadata:
        return docs[0].metadata["entity_name"]
    return "النظام الحالي"

def _classify_question(question: str) -> Literal["technical", "general", "inappropriate"]:
    """
    يستخدم مصنفًا سريعًا لتحديد نية المستخدم إلى ثلاث فئات.
    """
    if not classifier: raise RuntimeError("المصنف غير مهيأ.")
    
    perf_logger.start("routing")
    
    labels = [
        "سؤال فني أو استفسار عن معلومات محددة", 
        "تحية، شكر، أو سؤال عام عن الهوية مثل من أنت",
        "إهانة، كلام بذيء، أو عبارات عشوائية غير مفهومة"
    ]
    
    result = classifier(question, labels, multi_label=False)
    
    top_label = result['labels'][0]
    decision: Literal["technical", "general", "inappropriate"]
    if top_label == labels[0]:
        decision = "technical"
    elif top_label == labels[1]:
        decision = "general"
    else:
        decision = "inappropriate"
    
    perf_logger.end("routing", "N/A", question, {"decision": decision, "confidence": result['scores'][0]})
    logging.info(f"قرار التوجيه: '{decision}' (بثقة: {result['scores'][0]:.2f})")
    
    return decision

def _hybrid_retrieval_and_rerank(question: str, tenant_id: str, k: int) -> List[Document]:
    """تنفذ استراتيجية بحث هجينة ثم تعيد ترتيب النتائج."""
    if not vector_store or not cross_encoder: raise RuntimeError("مكونات البحث غير مهيأة.")
    
    perf_logger.start("retrieval_rerank")
    
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k * 5, 'filter': {'tenant_id': tenant_id}}
    )
    faiss_docs = faiss_retriever.invoke(question)

    tenant_docs = [doc for doc in all_docs_for_bm25 if doc.metadata.get("tenant_id") == tenant_id]
    bm25_docs = []
    if tenant_docs:
        corpus = [doc.page_content.split() for doc in tenant_docs]
        bm25 = BM25Okapi(corpus)
        tokenized_query = question.split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k * 5]
        bm25_docs = [tenant_docs[i] for i in top_indices]

    combined_docs = list({doc.page_content: doc for doc in faiss_docs + bm25_docs}.values())
    if not combined_docs:
        perf_logger.end("retrieval_rerank", tenant_id, question, {"status": "no_docs_found"})
        return []

    pairs = [[question, doc.page_content] for doc in combined_docs]
    scores = cross_encoder.predict(pairs)
    
    reranked_results = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True)
    
    final_docs = [doc for score, doc in reranked_results[:k]]
    
    perf_logger.end("retrieval_rerank", tenant_id, question, {"retrieved_count": len(final_docs)})
    logging.info(f"تم استرجاع وإعادة ترتيب {len(final_docs)} مستندًا ذا صلة.")
    
    return final_docs

# =================================================================================
# 6. نقطة الدخول الرئيسية (Main Entrypoint)
# =================================================================================

async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
    """
    الدالة الرئيسية التي تعالج سؤال المستخدم وتبث الإجابة بشكل تفاعلي.
    """
    if not rag_chain or not general_chain or not fallback_chain:
        raise RuntimeError("الوكيل غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")

    logging.info(f"استلام طلب جديد من العميل '{tenant_id}'.")
    
    try:
        category = _classify_question(question)
        tenant_name = _get_dynamic_identity(tenant_id)
        logging.info(f"الهوية الديناميكية المحددة: '{tenant_name}'")

        if category == "technical":
            logging.info("تنفيذ مسار الدعم الفني (RAG)...")
            relevant_docs = _hybrid_retrieval_and_rerank(question, tenant_id, k_results)
            
            if not relevant_docs:
                logging.warning("لم يتم العثور على مستندات ذات صلة. سيتم استخدام إجابة الطوارئ.")
                async for chunk in fallback_chain.astream({}):
                    yield chunk
                return

            async for chunk in rag_chain.astream({
                "question": question,
                "docs": relevant_docs,
                "tenant_name": tenant_name
            }):
                yield chunk
        
        elif category == "inappropriate":
            logging.info("تنفيذ مسار الرد على المدخلات غير الملائمة...")
            async for chunk in general_chain.astream({
                "question": question,
                "tenant_name": tenant_name
            }):
                yield chunk

        else: # category == "general"
            logging.info("تنفيذ مسار المحادثة العامة...")
            async for chunk in general_chain.astream({
                "question": question,
                "tenant_name": tenant_name
            }):
                yield chunk

    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}", exc_info=True)
        yield "عذرًا، حدث خطأ فني. فريقنا يعمل على إصلاحه."
        perf_logger.end("error", tenant_id, question, {"error": str(e)})
