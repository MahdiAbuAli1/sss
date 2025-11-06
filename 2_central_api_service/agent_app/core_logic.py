# import os
# import logging
# import asyncio
# import httpx
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage

# # --- vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ---
# # --- هذا هو القسم الذي يجب تعديله ---
# # --- vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ---

# # الاستيراد الصحيح للإصدارات الحديثة من LangChain
# # يتم استيراد كل وظيفة من مسارها الكامل والدقيق داخل الحزمة

# try:
#     from langchain.chains import create_history_aware_retriever
#     from langchain.chains.combine_documents import create_stuff_documents_chain
#     from langchain.chains import create_retrieval_chain
# except ImportError:
#     try:
#         from langchain.chains.history_aware_retriever import create_history_aware_retriever
#         from langchain.chains.combine_documents import create_stuff_documents_chain
#         from langchain.chains.retrieval import create_retrieval_chain
#     except ImportError:
#         # للإصدارات القديمة جداً
#         from langchain.chains import (
#             create_history_aware_retriever,
#             create_stuff_documents_chain,
#             create_retrieval_chain
#         )


# # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
# # --- نهاية القسم الذي يجب تعديله ---
# # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

# # استيراد مسجل الأداء
# from .performance_tracker import PerformanceLogger

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:4b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- متغيرات عالمية ---
# llm: Ollama = None
# vector_store: FAISS = None
# embeddings: OllamaEmbeddings = None
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()
# # --- إنشاء نسخة من مسجل الأداء ---
# perf_logger = PerformanceLogger()

# # --- 2. القوالب (لا تغيير هنا) ---
# REPHRASE_PROMPT = ChatPromptTemplate.from_template("""
# بالنظر إلى سجل المحادثة والسؤال الأخير، قم بصياغة سؤال مستقل يمكن فهمه بدون سجل المحادثة.
# سجل المحادثة: {chat_history}
# السؤال الأخير: {input}
# السؤال المستقل:""")

# ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
# - كن دائماً متعاوناً ومحترفاً.
# - إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
# - إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
# - لا تخترع إجابات أبداً. التزم بالسياق.

# السياق:
# {context}

# السؤال: {input}
# الإجابة:""")

# # --- 3. الدوال الأساسية (لا تغيير هنا) ---
# async def initialize_agent():
#     global llm, embeddings, vector_store
#     async with initialization_lock:
#         if vector_store is not None: return
#         logging.info("بدء تهيئة النماذج وقاعدة البيانات الموحدة...")
#         try:
#             async with httpx.AsyncClient( ) as client:
#                 await client.get(OLLAMA_HOST, timeout=10.0)
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة. يرجى تشغيل سكرت 'main_builder.py' أولاً.")

#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             logging.info("✅ الوكيل جاهز للعمل بقاعدة بيانات موحدة.")
#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# # --- 4. دالة للتحقق من جاهزية الوكيل ---
# def agent_ready() -> bool:
#     """التحقق من أن الوكيل جاهز للعمل"""
#     return vector_store is not None and llm is not None

# # --- 5. دالة get_answer_stream مع تسجيل الأداء ---
# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     """دالة رئيسية لتوليد الإجابات بشكل متدفق"""
#     question = request_info.get("question", "")
#     tenant_id = request_info.get("tenant_id", "default_session")
#     k_results = request_info.get("k_results", 4)
    
#     session_id = tenant_id or "default_session"

#     if not vector_store:
#         yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
#         return

#     perf_logger.start("total_request", tenant_id, question, {"k_results": k_results})

#     retriever = vector_store.as_retriever(
#         search_kwargs={'k': k_results, 'filter': {'tenant_id': tenant_id}}
#     )
    
#     user_chat_history = chat_history.get(session_id, [])

#     # --- بناء السلاسل ---
#     history_aware_retriever = create_history_aware_retriever(llm, retriever, REPHRASE_PROMPT)
#     document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
#     conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

#     logging.info(f"[{session_id}] بدء معالجة السؤال '{question}'...")
#     try:
#         full_answer = ""
#         # بدء تسجيل وقت تدفق الإجابة
#         perf_logger.start("llm_stream_generation", tenant_id, question)

#         async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
#             if "answer" in chunk and chunk["answer"] is not None:
#                 answer_chunk = chunk["answer"]
#                 full_answer += answer_chunk
#                 yield {"type": "chunk", "content": answer_chunk}
        
#         # إنهاء تسجيل وقت تدفق الإجابة
#         perf_logger.end("llm_stream_generation", tenant_id, question, {"answer_length": len(full_answer)})

#         # تحديث سجل المحادثة
#         user_chat_history.append(HumanMessage(content=question))
#         user_chat_history.append(AIMessage(content=full_answer))
#         chat_history[session_id] = user_chat_history[-10:] # الاحتفاظ بآخر 10 رسائل
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")
#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
#     finally:
#         # تسجيل إجمالي وقت الطلب في كل الحالات (نجاح أو فشل)
#         perf_logger.end("total_request", tenant_id, question)
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- النسخة النهائية المدمجة مع البحث الهجين ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- النسخة النهائية المصححة لمشكلة الاستيراد ---

# import os
# import logging
# import asyncio
# import httpx
# from typing import AsyncGenerator, Dict, List, cast

# from dotenv import load_dotenv
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.documents import Document

# # --- vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ---
# # --- هذا هو القسم الذي تم تعديله ---
# # --- vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ---

# # 1. إضافة استيرادات جديدة للبحث الهجين
# from langchain.retrievers import BM25Retriever, EnsembleRetriever

# # 2. استخدام المسارات الصحيحة والحديثة لوظائف السلاسل
# # هذا يحل مشكلة ImportError
# from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain

# # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
# # --- نهاية القسم الذي تم تعديله ---
# # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

# from .performance_tracker import PerformanceLogger

# # --- 1. الإعدادات (لا تغيير هنا ) ---
# # ... (بقية الكود يبقى كما هو دون أي تغيير) ...
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- متغيرات عالمية ---
# llm: Ollama = None
# ensemble_retriever: EnsembleRetriever = None 
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()
# perf_logger = PerformanceLogger()

# # --- 2. القوالب (لا تغيير هنا) ---
# REPHRASE_PROMPT = ChatPromptTemplate.from_template("""
# بالنظر إلى سجل المحادثة والسؤال الأخير، قم بصياغة سؤال مستقل يمكن فهمه بدون سجل المحادثة.
# سجل المحادثة: {chat_history}
# السؤال الأخير: {input}
# السؤال المستقل:""")

# ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
# - كن دائماً متعاوناً ومحترفاً.
# - إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
# - إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
# - لا تخترع إجابات أبداً. التزم بالسياق.

# السياق:
# {context}

# السؤال: {input}
# الإجابة:""")

# # --- 3. الدوال الأساسية (لا تغيير هنا) ---
# def _load_all_docs_from_faiss(vector_store: FAISS) -> List[Document]:
#     return list(cast(dict, vector_store.docstore._dict).values())

# async def initialize_agent():
#     global llm, ensemble_retriever
#     async with initialization_lock:
#         if ensemble_retriever is not None: return
#         logging.info("بدء تهيئة النماذج والمسترجع الهجين...")
#         try:
#             async with httpx.AsyncClient( ) as client:
#                 await client.get(OLLAMA_HOST, timeout=10.0)
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
            
#             logging.info("تحميل قاعدة بيانات FAISS...")
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة. يرجى تشغيل 'main_builder.py' أولاً.")
            
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
#             faiss_vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={'k': 4})
#             logging.info("✅ تم تحميل المسترجع الدلالي (FAISS).")

#             logging.info("بناء مسترجع الكلمات المفتاحية (BM25)...")
#             all_docs = await asyncio.to_thread(_load_all_docs_from_faiss, faiss_vector_store)
#             bm25_retriever = BM25Retriever.from_documents(all_docs)
#             bm25_retriever.k = 4
#             logging.info("✅ تم بناء المسترجع (BM25).")

#             ensemble_retriever = EnsembleRetriever(
#                 retrievers=[bm25_retriever, faiss_retriever],
#                 weights=[0.5, 0.5]
#             )
#             logging.info("🚀 الوكيل جاهز للعمل مع المسترجع الهجين.")

#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# # --- 4. دالة للتحقق من جاهزية الوكيل (لا تغيير هنا) ---
# def agent_ready() -> bool:
#     return ensemble_retriever is not None and llm is not None

# # --- 5. دالة get_answer_stream (لا تغيير هنا) ---
# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     question = request_info.get("question", "")
#     tenant_id = request_info.get("tenant_id", "default_session")
    
#     session_id = tenant_id or "default_session"

#     if not ensemble_retriever:
#         yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
#         return

#     perf_logger.start("total_request", tenant_id, question, {"retriever_type": "hybrid"})
    
#     user_chat_history = chat_history.get(session_id, [])

#     history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, REPHRASE_PROMPT)
#     document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
#     conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

#     logging.info(f"[{session_id}] بدء معالجة السؤال '{question}'...")
#     try:
#         full_answer = ""
#         perf_logger.start("llm_stream_generation", tenant_id, question)

#         async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
#             if "answer" in chunk and chunk["answer"] is not None:
#                 answer_chunk = chunk["answer"]
#                 full_answer += answer_chunk
#                 yield {"type": "chunk", "content": answer_chunk}
        
#         perf_logger.end("llm_stream_generation", tenant_id, question, {"answer_length": len(full_answer)})

#         user_chat_history.append(HumanMessage(content=question))
#         user_chat_history.append(AIMessage(content=full_answer))
#         chat_history[session_id] = user_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")
#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
#     finally:
#         perf_logger.end("total_request", tenant_id, question)


# المسار: 2_central_api_service/agent_app/core_logic.py
# --- النسخة النهائية مع سلسلة RAG المتقدمة (إعادة صياغة + بحث هجين) ---
#الاصدار الثاني
# import os
# import logging
# import asyncio
# import httpx
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# from .performance_tracker import PerformanceLogger

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- 2. الملفات الشخصية للأنظمة (فكرتك الرائعة!) ---
# SYSTEM_PROFILES = {
#     "sys": {
#         "name": "نظام إدارة طلبات الاعتماد",
#         "description": "نظام لتتبع مراحل الحصول على الاعتماد من التقديم حتى إصدار الشهادة.",
#         "keywords": ["طلب اعتماد", "قوائم التحقق", "دراسة مكتبية", "زيارة ميدانية", "إجراءات تصحيحية"]
#     },
#     "university_alpha": {
#         "name": "تطبيق Plant Care",
#         "description": "تطبيق ذكي لمساعدة المزارعين في التعرف على الآفات الزراعية.",
#         "keywords": ["متطلبات وظيفية", "حالات استخدام", "تصميم النظام", "مخطط علاقات", "plant care"]
#     },
#     "school_beta": {
#         "name": "مستندات الشبكات العصبية",
#         "description": "مجموعة من المستندات التعليمية حول الشبكات العصبية التلافيفية (CNN) ومكتبة TensorFlow.",
#         "keywords": ["شبكة عصبية", "tensorflow", "convolutional layer", "relu", "pooling"]
#     },
#     "un": {
#         "name": "بوابة المشتريات الإلكترونية للأمم المتحدة",
#         "description": "دليل استخدام نظام الشراء الإلكتروني الخاص بمكتب الأمم المتحدة لخدمات المشاريع (UNOPS).",
#         "keywords": ["مناقصات", "تسجيل الدخول", "عطاءات", "unops", "esourcing"]
#     }
# }

# # --- 3. القوالب المتقدمة ---
# REWRITE_PROMPT_TEMPLATE = """
# أنت خبير في النظام التالي:
# - اسم النظام: {system_name}
# - وصفه: {system_description}
# - مصطلحات هامة: {system_keywords}

# مهمتك هي تحويل سؤال المستخدم العام إلى استعلام بحث دقيق ومحدد لاستخدامه في قاعدة بيانات تقنية. استخدم المصطلحات الهامة لإنشاء أفضل استعلام ممكن.

# سؤال المستخدم: {question}

# الاستعلام المحسّن:"""

# REPHRASE_HISTORY_PROMPT = ChatPromptTemplate.from_template("""
# بالنظر إلى سجل المحادثة والسؤال الأخير، قم بصياغة سؤال مستقل يمكن فهمه بدون سجل المحادثة.
# سجل المحادثة: {chat_history}
# السؤال الأخير: {input}
# السؤال المستقل:""")

# ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
# - كن دائماً متعاوناً ومحترفاً.
# - إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
# - إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
# - لا تخترع إجابات أبداً. التزم بالسياق.

# السياق:
# {context}

# السؤال: {input}
# الإجابة:""")

# # --- 4. المتغيرات العالمية ---
# llm: Ollama = None
# vector_store: FAISS = None
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()
# perf_logger = PerformanceLogger()

# # --- 5. الدوال الأساسية ---

# def _load_all_docs_from_faiss(vs: FAISS) -> List[Document]:
#     return list(vs.docstore._dict.values())

# async def initialize_agent():
#     global llm, vector_store
#     async with initialization_lock:
#         if vector_store is not None: return
#         logging.info("بدء تهيئة النماذج وقاعدة البيانات الموحدة...")
#         try:
#             async with httpx.AsyncClient( ) as client:
#                 await client.get(OLLAMA_HOST, timeout=10.0)
            
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError("قاعدة البيانات الموحدة غير موجودة.")

#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             logging.info("✅ الوكيل جاهز للعمل بقاعدة بيانات موحدة.")
#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return vector_store is not None and llm is not None

# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     question = request_info.get("question", "")
#     tenant_id = request_info.get("tenant_id", "default_session")
#     k_results = request_info.get("k_results", 8)
#     session_id = tenant_id or "default_session"

#     if not agent_ready():
#         yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
#         return

#     perf_logger.start("total_request", tenant_id, question)
#     user_chat_history = chat_history.get(session_id, [])

#     try:
#         # --- المرحلة 1: إعادة صياغة السؤال بناءً على السياق (فكرتك!) ---
#         profile = SYSTEM_PROFILES.get(tenant_id, {})
#         if profile:
#             logging.info(f"[{session_id}] استخدام ملف شخصي لإعادة صياغة السؤال للعميل '{tenant_id}'.")
#             rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
#             rewriter_chain = rewrite_prompt | llm | StrOutputParser()
            
#             # هذه هي الخطوة التي قد تكون بطيئة
#             effective_question = await rewriter_chain.ainvoke({
#                 "system_name": profile.get("name", ""),
#                 "system_description": profile.get("description", ""),
#                 "system_keywords": ", ".join(profile.get("keywords", [])),
#                 "question": question
#             })
#             logging.info(f"[{session_id}] السؤال الأصلي: '{question}' -> السؤال المحسّن: '{effective_question}'")
#         else:
#             effective_question = question
#             logging.warning(f"[{session_id}] لم يتم العثور على ملف شخصي للعميل '{tenant_id}'. سيتم استخدام السؤال الأصلي.")

#         # --- المرحلة 2: بناء المسترجع الهجين المفلتر ---
#         all_docs = _load_all_docs_from_faiss(vector_store)
#         tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]

#         if not tenant_docs:
#             yield {"type": "error", "content": f"لا توجد بيانات للعميل '{tenant_id}'."}
#             return

#         bm25_retriever = BM25Retriever.from_documents(tenant_docs)
#         bm25_retriever.k = k_results // 2
        
#         faiss_retriever = vector_store.as_retriever(
#             search_kwargs={'k': k_results // 2, 'filter': {'tenant_id': tenant_id}}
#         )
        
#         ensemble_retriever = EnsembleRetriever(
#             retrievers=[bm25_retriever, faiss_retriever],
#             weights=[0.5, 0.5]
#         )

#         # --- المرحلة 3: بناء سلسلة RAG الكاملة ---
#         history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, REPHRASE_HISTORY_PROMPT)
#         document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
#         conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

#         # --- المرحلة 4: التنفيذ والبث ---
#         logging.info(f"[{session_id}] بدء معالجة السؤال '{effective_question}'...")
#         full_answer = ""
#         perf_logger.start("llm_stream_generation", tenant_id, question)

#         async for chunk in conversational_rag_chain.astream({"input": effective_question, "chat_history": user_chat_history}):
#             if "answer" in chunk and chunk["answer"] is not None:
#                 answer_chunk = chunk["answer"]
#                 full_answer += answer_chunk
#                 yield {"type": "chunk", "content": answer_chunk}
        
#         perf_logger.end("llm_stream_generation", tenant_id, question)

#         # تحديث سجل المحادثة
#         user_chat_history.append(HumanMessage(content=question)) # نحفظ السؤال الأصلي
#         user_chat_history.append(AIMessage(content=full_answer))
#         chat_history[session_id] = user_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
#     finally:
#         perf_logger.end("total_request", tenant_id, question)



# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 11.0: النسخة النهائية مع قالب البساطة المطلقة ---
#الصور المرسله لرمزي هي نتيجه لهذا 
# import os
# import logging
# import asyncio
# import httpx
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# from flashrank import Ranker, RerankRequest

# from .performance_tracker import PerformanceLogger

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- 2. الملفات الشخصية للأنظمة ---
# SYSTEM_PROFILES = {
#     "sys": {
#         "name": "نظام إدارة طلبات الاعتماد",
#         "description": "نظام إلكتروني لتتبع رحلة الحصول على الاعتماد، بدءًا من إنشاء الحساب، تقديم الطلب، دفع الفواتير، مرورًا بمراحل التقييم والزيارات الميدانية، وانتهاءً باتخاذ القرار وإصدار الشهادة.",
#         "keywords": ["إنشاء حساب", "تسجيل الدخول", "طلب اعتماد جديد", "قوائم التحقق", "دراسة مكتبية", "زيارة ميدانية", "إجراءات تصحيحية", "فاتورة", "شهادة الاعتماد"]
#     },
#     "university_alpha": {
#         "name": "تطبيق Plant Care الزراعي",
#         "description": "تطبيق ذكي لمساعدة المزارعين في تشخيص أمراض النباتات والآفات الزراعية باستخدام الذكاء الاصطناعي، مع التركيز على محصولي القات والعنب.",
#         "keywords": ["تشخيص النبات", "الآفات الزراعية", "متطلبات وظيفية", "حالات استخدام", "تصميم النظام", "plant care", "الذكاء الاصطناعي في الزراعة"]
#     },
#     "school_beta": {
#         "name": "مستندات الشبكات العصبية",
#         "description": "مادة تعليمية تشرح مفاهيم الشبكات العصبية، مكتبة TensorFlow، والشبكات التلافيفية (CNN)، بما في ذلك الطبقات، دوال التنشيط، وخوارزميات التحسين.",
#         "keywords": ["شبكة عصبية", "tensorflow", "convolutional layer", "relu", "pooling", "dense layer", "loss function", "optimizer", "backpropagation"]
#     },
#     "un": {
#         "name": "بوابة المشتريات الإلكترونية للأمم المتحدة (UNOPS eSourcing)",
#         "description": "دليل إرشادي للموردين حول كيفية استخدام نظام الشراء الإلكتروني الخاص بمكتب الأمم المتحدة لخدمات المشاريع (UNOPS)، ويشمل التسجيل، البحث عن المناقصات، وتقديم العطاءات.",
#         "keywords": ["مناقصات", "تسجيل الدخول", "تقديم العطاءات", "unops", "esourcing", "ungm.org", "موردين", "حالة المناقصة"]
#     }
# }

# # --- 3. القالب النهائي لإعادة الصياغة (الإصدار 11.0: البساطة المطلقة) ---
# REWRITE_PROMPT_TEMPLATE = """
# مهمتك واضحة ومحددة: حول سؤال المستخدم إلى جملة بحث قصيرة ومركزة.

# **سياق النظام:**
# - اسم النظام: {system_name}
# - وصفه: {system_description}
# - مصطلحات هامة: {system_keywords}

# ---
# **قواعد صارمة لا يمكن كسرها:**
# 1.  **الناتج جملة واحدة فقط:** يجب أن يكون الناتج جملة قصيرة وموجزة.
# 2.  **التركيز على النية:** استخدم الكلمات الأساسية من سؤال المستخدم والمصطلحات الهامة لبناء جملة تعبر عن القصد.
# 3.  **إذا كان السؤال عن تعريف النظام:** (مثل "ما هو هذا النظام؟")، يجب أن يكون الناتج "وصف {system_name}".
# 4.  **إذا كان السؤال خارج السياق تمامًا:** (مثل "من هو ميسي؟")، **أعد السؤال الأصلي كما هو بالضبط.**
# 5.  **ممنوع الشرح:** لا تقم أبدًا بشرح الاستعلام أو إضافة أي نص إضافي. الناتج هو جملة البحث فقط.

# ---
# **أمثلة للتنفيذ الصحيح:**

# سؤال المستخدم: ماهو هذا النظام؟
# الاستعلام المحسّن: وصف نظام إدارة طلبات الاعتماد

# سؤال المستخدم: كيف اضيف حساب جديد؟
# الاستعلام المحسّن: خطوات إضافة حساب جديد في نظام إدارة طلبات الاعتماد

# سؤال المستخدم: من هي جورجينا؟
# الاستعلام المحسّن: من هي جورجينا؟
# ---

# **المهمة المطلوبة:**

# سؤال المستخدم: {question}

# الاستعلام المحسّن:
# """

# # --- باقي القوالب ---
# ANSWER_PROMPT = ChatPromptTemplate.from_template("أنت \"مرشد الدعم\"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على \"السياق\" المقدم.\n- كن دائماً متعاوناً ومحترفاً.\n- إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.\n- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: \"بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال.\"\n- لا تخترع إجابات أبداً. التزم بالسياق.\n\nالسياق:\n{context}\n\nالسؤال: {input}\nالإجابة:")

# # --- 4. المتغيرات العالمية ---
# llm: Ollama = None
# vector_store: FAISS = None
# reranker: Ranker = None
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()
# perf_logger = PerformanceLogger()

# # --- 5. الدوال الأساسية ---

# def _load_all_docs_from_faiss(vs: FAISS) -> List[Document]:
#     return list(vs.docstore._dict.values())

# def _clean_rewritten_query(raw_query: str) -> str:
#     lines = raw_query.strip().split('\n')
#     for line in reversed(lines):
#         cleaned_line = line.strip()
#         if cleaned_line:
#             if cleaned_line.startswith("الاستعلام المحسّن:"):
#                 return cleaned_line.replace("الاستعلام المحسّن:", "").strip()
#             return cleaned_line
#     return raw_query

# async def initialize_agent():
#     global llm, vector_store, reranker
#     async with initialization_lock:
#         if vector_store is not None: return
#         logging.info("بدء تهيئة النماذج وقاعدة البيانات و Reranker...")
#         try:
#             async with httpx.AsyncClient( ) as client:
#                 await client.get(OLLAMA_HOST, timeout=10.0)
            
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError("قاعدة البيانات الموحدة غير موجودة.")

#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
            
#             reranker = Ranker()
            
#             logging.info("✅ الوكيل جاهز للعمل (مع Reranker).")
#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return vector_store is not None and llm is not None and reranker is not None

# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     question = request_info.get("question", "")
#     tenant_id = request_info.get("tenant_id", "default_session")
#     k_results = request_info.get("k_results", 10)
#     session_id = tenant_id or "default_session"

#     if not agent_ready():
#         yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
#         return

#     user_chat_history = chat_history.get(session_id, [])

#     try:
#         effective_question = question
#         profile = SYSTEM_PROFILES.get(tenant_id)
        
#         if profile:
#             logging.info(f"[{session_id}] استخدام ملف شخصي متقدم لإعادة صياغة السؤال...")
#             rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
#             rewriter_chain = rewrite_prompt | llm | StrOutputParser()
            
#             # هنا لا يوجد متغير {الفعل}، لذا لن يحدث الخطأ
#             raw_rewritten_query = await rewriter_chain.ainvoke({
#                 "system_name": profile.get("name", ""),
#                 "system_description": profile.get("description", ""),
#                 "system_keywords": ", ".join(profile.get("keywords", [])),
#                 "question": question
#             })
            
#             effective_question = _clean_rewritten_query(raw_rewritten_query)
#             logging.info(f"[{session_id}] السؤال الأصلي: '{question}' -> السؤال المحسّن: '{effective_question}'")

#         all_docs = _load_all_docs_from_faiss(vector_store)
#         tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]

#         if not tenant_docs:
#             yield {"type": "error", "content": f"لا توجد بيانات للعميل '{tenant_id}'."}
#             return

#         bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=k_results)
#         faiss_retriever = vector_store.as_retriever(
#             search_kwargs={'k': k_results, 'filter': {'tenant_id': tenant_id}}
#         )
#         ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
#         logging.info(f"[{session_id}] بدء الاسترجاع الأولي لـ '{effective_question}'...")
#         initial_docs = await ensemble_retriever.ainvoke(effective_question)
#         logging.info(f"[{session_id}] تم استرجاع {len(initial_docs)} مستند أولي.")

#         logging.info(f"[{session_id}] بدء إعادة الترتيب والفلترة...")
        
#         passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(initial_docs)]
        
#         rerank_request = RerankRequest(query=question, passages=passages)
#         all_reranked_results = reranker.rerank(rerank_request)
#         top_4_results = all_reranked_results[:4]
        
#         original_docs_map = {doc.page_content: doc for doc in initial_docs}
#         reranked_docs = [original_docs_map[res["text"]] for res in top_4_results if res["text"] in original_docs_map]
        
#         logging.info(f"[{session_id}] تم فلترة المستندات إلى {len(reranked_docs)} مستند عالي الصلة.")

#         document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        
#         logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
#         full_answer = ""
        
#         async for chunk in document_chain.astream({"input": question, "context": reranked_docs, "chat_history": user_chat_history}):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}

#         user_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
#         chat_history[session_id] = user_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}


# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 12.0: النسخة النهائية مع قالب استخراج الكلمات المفتاحية ---

# import os
# import logging
# import asyncio
# import httpx
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.chains.combine_documents import create_stuff_documents_chain

# from flashrank import Ranker, RerankRequest

# from .performance_tracker import PerformanceLogger

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- 2. الملفات الشخصية للأنظمة ---
# SYSTEM_PROFILES = {
#     "sys": {
#         "name": "نظام إدارة طلبات الاعتماد",
#         "description": "نظام إلكتروني لتتبع رحلة الحصول على الاعتماد.",
#         "keywords": ["إنشاء حساب", "تسجيل الدخول", "طلب اعتماد", "قوائم التحقق", "دراسة مكتبية", "زيارة ميدانية", "إجراءات تصحيحية", "فاتورة", "شهادة"]
#     },
#     "university_alpha": {
#         "name": "تطبيق Plant Care",
#         "description": "تطبيق ذكي لتشخيص أمراض النباتات والآفات الزراعية.",
#         "keywords": ["تشخيص النبات", "آفات زراعية", "متطلبات وظيفية", "حالات استخدام", "تصميم النظام", "plant care"]
#     },
#     "school_beta": {
#         "name": "مستندات الشبكات العصبية",
#         "description": "مادة تعليمية عن الشبكات العصبية و TensorFlow.",
#         "keywords": ["شبكة عصبية", "tensorflow", "cnn", "layer", "relu", "pooling", "optimizer"]
#     },
#     "un": {
#         "name": "بوابة المشتريات الإلكترونية للأمم المتحدة",
#         "description": "دليل إرشادي للموردين لاستخدام نظام الشراء الإلكتروني.",
#         "keywords": ["مناقصات", "تسجيل الدخول", "عطاءات", "unops", "esourcing", "ungm.org", "موردين"]
#     }
# }

# # --- 3. القالب النهائي (الإصدار 12.0: استخراج الكلمات المفتاحية) ---
# REWRITE_PROMPT_TEMPLATE = """
# مهمتك هي استخراج الكلمات المفتاحية الأكثر أهمية من سؤال المستخدم لتحسين البحث.

# **سياق النظام:** {system_name}
# **مصطلحات هامة:** {system_keywords}

# ---
# **القواعد:**
# 1.  **إذا كان السؤال عامًا عن النظام** (مثل "ما هو هذا النظام؟")، أرجع اسم النظام فقط: `{system_name}`.
# 2.  **إذا كان السؤال عن خطوات أو كيفية فعل شيء** (مثل "كيف أضيف مستخدم؟")، أرجع الفعل والمفعول به: `إضافة مستخدم جديد`.
# 3.  **إذا كان السؤال عن تعريف مصطلح** (مثل "ماهي الشبكات العصبية؟")، أرجع المصطلح نفسه: `الشبكات العصبية`.
# 4.  **إذا كان السؤال خارج السياق تمامًا** (مثل "من هو ميسي؟")، أرجع السؤال الأصلي كما هو.
# 5.  **الناتج يجب أن يكون قصيرًا جدًا ومباشرًا.** لا تستخدم جمل كاملة.

# ---
# **أمثلة:**

# سؤال المستخدم: ماهو هذا النظام باختصار
# الاستعلام المحسّن: نظام إدارة طلبات الاعتماد

# سؤال المستخدم: كيفيه الوصول للنظام
# الاستعلام المحسّن: كيفية تسجيل الدخول

# سؤال المستخدم: ماهي الشبكات العصبيه
# الاستعلام المحسّن: الشبكات العصبية

# سؤال المستخدم: من هي جورجينا
# الاستعلام المحسّن: من هي جورجينا
# ---

# **المهمة المطلوبة:**

# سؤال المستخدم: {question}

# الاستعلام المحسّن:
# """

# # --- باقي القوالب ---
# ANSWER_PROMPT = ChatPromptTemplate.from_template("أنت \"مرشد الدعم\"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على \"السياق\" المقدم.\n- كن دائماً متعاوناً ومحترفاً.\n- إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.\n- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: \"بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال.\"\n- لا تخترع إجابات أبداً. التزم بالسياق.\n\nالسياق:\n{context}\n\nالسؤال: {input}\nالإجابة:")

# # --- 4. المتغيرات العالمية ---
# llm: Ollama = None
# vector_store: FAISS = None
# reranker: Ranker = None
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()
# perf_logger = PerformanceLogger()

# # --- 5. الدوال الأساسية (معظمها يبقى كما هو) ---

# def _load_all_docs_from_faiss(vs: FAISS) -> List[Document]:
#     return list(vs.docstore._dict.values())

# def _clean_rewritten_query(raw_query: str) -> str:
#     lines = raw_query.strip().split('\n')
#     for line in reversed(lines):
#         cleaned_line = line.strip()
#         if cleaned_line:
#             if cleaned_line.startswith("الاستعلام المحسّن:"):
#                 return cleaned_line.replace("الاستعلام المحسّن:", "").strip()
#             return cleaned_line
#     return raw_query.strip()

# async def initialize_agent():
#     global llm, vector_store, reranker
#     async with initialization_lock:
#         if vector_store is not None: return
#         logging.info("بدء تهيئة النماذج وقاعدة البيانات و Reranker...")
#         try:
#             async with httpx.AsyncClient( ) as client:
#                 await client.get(OLLAMA_HOST, timeout=10.0)
            
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError("قاعدة البيانات الموحدة غير موجودة.")

#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
            
#             reranker = Ranker()
            
#             logging.info("✅ الوكيل جاهز للعمل (مع Reranker).")
#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return vector_store is not None and llm is not None and reranker is not None

# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     question = request_info.get("question", "")
#     tenant_id = request_info.get("tenant_id", "default_session")
#     k_results = request_info.get("k_results", 10)
#     session_id = tenant_id or "default_session"

#     if not agent_ready():
#         yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
#         return

#     user_chat_history = chat_history.get(session_id, [])

#     try:
#         effective_question = question
#         profile = SYSTEM_PROFILES.get(tenant_id)
        
#         if profile:
#             logging.info(f"[{session_id}] استخدام ملف شخصي متقدم لإعادة صياغة السؤال...")
#             rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
#             rewriter_chain = rewrite_prompt | llm | StrOutputParser()
            
#             raw_rewritten_query = await rewriter_chain.ainvoke({
#                 "system_name": profile.get("name", ""),
#                 "system_description": profile.get("description", ""),
#                 "system_keywords": ", ".join(profile.get("keywords", [])),
#                 "question": question
#             })
            
#             effective_question = _clean_rewritten_query(raw_rewritten_query)
#             logging.info(f"[{session_id}] السؤال الأصلي: '{question}' -> الاستعلام المحسّن: '{effective_question}'")

#         all_docs = _load_all_docs_from_faiss(vector_store)
#         tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]

#         if not tenant_docs:
#             yield {"type": "error", "content": f"لا توجد بيانات للعميل '{tenant_id}'."}
#             return

#         bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=k_results)
#         faiss_retriever = vector_store.as_retriever(
#             search_kwargs={'k': k_results, 'filter': {'tenant_id': tenant_id}}
#         )
#         ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
#         logging.info(f"[{session_id}] بدء الاسترجاع الأولي لـ '{effective_question}'...")
#         initial_docs = await ensemble_retriever.ainvoke(effective_question)
#         logging.info(f"[{session_id}] تم استرجاع {len(initial_docs)} مستند أولي.")

#         logging.info(f"[{session_id}] بدء إعادة الترتيب والفلترة...")
        
#         passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(initial_docs)]
        
#         rerank_request = RerankRequest(query=question, passages=passages)
#         all_reranked_results = reranker.rerank(rerank_request)
#         top_4_results = all_reranked_results[:4]
        
#         original_docs_map = {doc.page_content: doc for doc in initial_docs}
#         reranked_docs = [original_docs_map[res["text"]] for res in top_4_results if res["text"] in original_docs_map]
        
#         logging.info(f"[{session_id}] تم فلترة المستندات إلى {len(reranked_docs)} مستند عالي الصلة.")

#         document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        
#         logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
#         full_answer = ""
        
#         async for chunk in document_chain.astream({"input": question, "context": reranked_docs, "chat_history": user_chat_history}):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}

#         user_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
#         chat_history[session_id] = user_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}





#تطبيق المسترجع الشامل 
# المسار: 2_central_api_service/agent_app/agent_logic.py (اسم مقترح للملف المحدث)
##Hybrid + Parent + Reranke
#بطي بلنه يثوم ببناء الخمسترؤجعات مسبقا بحي3ث يسهل ويسرع عمليه البحث 
# import os
# import logging
# import asyncio
# import pickle
# import time
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.storage import InMemoryStore
# from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from flashrank import Ranker, RerankRequest

# # --- 1. الإعدادات (مع إضافة مسارات الذاكرة المؤقتة) ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL_NAME", "qwen2:1.5b-instruct-q4_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
# CACHE_DIR = os.path.join(PROJECT_ROOT, "3_shared_resources", "retriever_cache") # <-- مسار الذاكرة المؤقتة
# TOP_K = 7

# # --- 2. قوالب التوجيه (Prompts) ---

# # قالب "حارس البوابة" لتصنيف الأسئلة
# QUESTION_CLASSIFIER_PROMPT = """
# Your task is to classify the user's question into one of three categories: "specific_query", "general_chitchat", or "nonsensical".
# - "specific_query": The user is asking a specific question that can likely be answered from a knowledge base (e.g., "how do I reset my password?", "what is max pooling?").
# - "general_chitchat": The user is asking a general knowledge question or making a greeting (e.g., "hello", "who is the president?", "what is the weather?").
# - "nonsensical": The user's input is random characters, gibberish, or makes no sense (e.g., "asdfgh", "blablabla", "qwertyy").

# User Question: "{question}"
# Category:
# """

# # قالب الإجابة الديناميكي
# DYNAMIC_PROMPT_TEMPLATE = """
# أنت "مساعد الدعم الذكي". مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصريًا** على "السياق" المقدم لك من قاعدة المعرفة.

# **قواعد صارمة:**
# 1.  **التحية دائمًا:** ابدأ إجابتك بعبارة ترحيبية مناسبة.
# 2.  **الالتزام المطلق بالسياق:** إذا كانت المعلومات غير موجودة، قل **فقط**: "لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
# 3.  **التكيف مع مستوى التفصيل المطلوب ({verbosity}):**
#     - **"مختصر"**: قدم إجابة موجزة في جملة أو جملتين.
#     - **"مفصل"**: قدم إجابة شاملة ومنظمة باستخدام القوائم.
# 4.  **الخاتمة التفاعلية:** اختتم دائمًا بسؤال تفاعلي، مثل: "هل هناك أي شيء آخر يمكنني مساعدتك به؟".

# ---
# **السياق:**
# {context}
# ---
# **سؤال المستخدم:** {question}
# ---
# **مستوى التفصيل المطلوب:** {verbosity}
# ---
# **إجابتك:**
# """

# # --- 3. فئة إدارة المسترجعات (مع التخزين المؤقت) ---
# class RetrieverManager:
#     def __init__(self, vector_store: FAISS):
#         self._vector_store = vector_store
#         self._cache = self._load_or_build_cache()

#     def _load_or_build_cache(self) -> Dict:
#         os.makedirs(CACHE_DIR, exist_ok=True)
#         cache_file = os.path.join(CACHE_DIR, "retriever_cache.pkl")
        
#         if os.path.exists(cache_file):
#             print("🧠 مدير المسترجعات: تحميل المسترجعات من الذاكرة المؤقتة (Cache)...")
#             with open(cache_file, "rb") as f:
#                 return pickle.load(f)
        
#         print("⚠️ مدير المسترجعات: الذاكرة المؤقتة غير موجودة. بدء عملية البناء (قد تستغرق وقتًا)...")
#         all_docs = list(self._vector_store.docstore._dict.values())
        
#         all_tenant_docs: Dict[str, List[Document]] = {}
#         for doc in all_docs:
#             tenant_id = doc.metadata.get("tenant_id")
#             if tenant_id:
#                 if tenant_id not in all_tenant_docs:
#                     all_tenant_docs[tenant_id] = []
#                 all_tenant_docs[tenant_id].append(doc)

#         new_cache = {}
#         for tenant_id, docs in all_tenant_docs.items():
#             print(f"   -> بناء مسترجعات للعميل: {tenant_id}")
#             new_cache[tenant_id] = {}
            
#             new_cache[tenant_id]['bm25'] = BM25Retriever.from_documents(docs)
            
#             store = InMemoryStore()
#             parent_retriever = ParentDocumentRetriever(
#                 vectorstore=self._vector_store, 
#                 docstore=store, 
#                 child_splitter=RecursiveCharacterTextSplitter(chunk_size=400)
#             )
#             parent_retriever.add_documents(docs, ids=None)
#             new_cache[tenant_id]['parent'] = parent_retriever
        
#         with open(cache_file, "wb") as f:
#             pickle.dump(new_cache, f)
#         print("✅ مدير المسترجعات: اكتمل بناء وحفظ الذاكرة المؤقتة.")
#         return new_cache

#     def get_retrievers(self, tenant_id: str) -> Dict:
#         if tenant_id not in self._cache:
#             raise ValueError(f"لا توجد مسترجعات مخزنة للعميل: {tenant_id}")
        
#         bm25_retriever = self._cache[tenant_id]['bm25']
#         parent_retriever = self._cache[tenant_id]['parent']
#         faiss_retriever = self._vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
        
#         hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
#         return {"hybrid": hybrid_retriever, "parent": parent_retriever}

# # --- 4. المتغيرات العالمية المهيكلة ---
# llm_answer: Ollama = None
# llm_classifier: Ollama = None
# vector_store: FAISS = None
# reranker: Ranker = None
# retriever_manager: RetrieverManager = None
# chat_history: Dict[str, List] = {}
# initialization_lock = asyncio.Lock()

# # --- 5. دوال التهيئة والتحقق ---
# async def initialize_agent():
#     global llm_answer, llm_classifier, vector_store, reranker, retriever_manager
#     async with initialization_lock:
#         if retriever_manager is not None: return
        
#         logging.info("--- 🚀 بدء تهيئة العقل الذكي (v-Final) ---")
#         try:
#             # تهيئة النماذج
#             llm_answer = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
#             llm_classifier = Ollama(model=CLASSIFIER_MODEL, base_url=OLLAMA_HOST)
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
#             # تهيئة أدوات الاسترجاع
#             vector_store = await asyncio.to_thread(FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
#             reranker = Ranker()
            
#             # تهيئة مدير المسترجعات (مع التخزين المؤقت)
#             retriever_manager = RetrieverManager(vector_store)
            
#             logging.info("--- ✅ العقل الذكي جاهز للعمل ---")
#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return retriever_manager is not None

# # --- 6. الدالة الرئيسية لتوليد الإجابة (معادة الهيكلة بالكامل) ---

# def _get_verbosity(question: str) -> str:
#     """يحدد مستوى التفصيل المطلوب."""
#     question_lower = question.lower()
#     if any(word in question_lower for word in ["باختصار", "موجز"]):
#         return "مختصر"
#     return "مفصل"

# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     question = request_info.get("question", "")
#     tenant_id = request_info.get("tenant_id", "default_session")
#     session_id = tenant_id or "default_session"

#     if not agent_ready():
#         yield {"type": "error", "content": "الوكيل غير جاهز بعد، يرجى المحاولة بعد قليل."}
#         return

#     try:
#         # --- المرحلة 1: حارس البوابة (التصنيف المسبق) ---
#         classifier_prompt = ChatPromptTemplate.from_template(QUESTION_CLASSIFIER_PROMPT)
#         classifier_chain = classifier_prompt | llm_classifier | StrOutputParser()
#         classification_result = await classifier_chain.ainvoke({"question": question})
#         classification = classification_result.strip().lower()
        
#         if "general_chitchat" in classification:
#             yield {"type": "full_answer", "content": "أنا مساعد متخصص ولا أستطيع الإجابة على أسئلة عامة. هل لديك سؤال حول النظام؟"}
#             return
#         if "nonsensical" in classification:
#             yield {"type": "full_answer", "content": "لم أفهم سؤالك. هل يمكنك إعادة صياغته؟"}
#             return

#         # --- المرحلة 2: الاسترجاع الشامل (فائق السرعة) ---
#         retrievers = retriever_manager.get_retrievers(tenant_id)
#         hybrid_retriever = retrievers['hybrid']
#         parent_retriever = retrievers['parent']
        
#         hybrid_docs, parent_docs = await asyncio.gather(
#             hybrid_retriever.ainvoke(question),
#             asyncio.to_thread(parent_retriever.invoke, question)
#         )
        
#         combined_docs = hybrid_docs + parent_docs
#         unique_docs = list({doc.page_content: doc for doc in reversed(combined_docs)}.values())[::-1]

#         if not unique_docs:
#             yield {"type": "full_answer", "content": "لم يتم العثور على أي معلومات ذات صلة في قاعدة المعرفة."}
#             return

#         # --- المرحلة 3: إعادة الترتيب (Reranking) ---
#         passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(unique_docs)]
#         reranked_results = reranker.rerank(RerankRequest(query=question, passages=passages))
#         top_results = reranked_results[:4]
        
#         original_docs_map = {i: doc for i, doc in enumerate(unique_docs)}
#         final_context_docs = [original_docs_map[res["id"]] for res in top_results]
#         final_context = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])

#         # --- المرحلة 4: توليد الإجابة النهائية (Streaming) ---
#         answer_prompt = ChatPromptTemplate.from_template(DYNAMIC_PROMPT_TEMPLATE)
#         answer_chain = answer_prompt | llm_answer | StrOutputParser()
#         verbosity = _get_verbosity(question)
        
#         full_answer = ""
#         async for chunk in answer_chain.astream({
#             "context": final_context,
#             "question": question,
#             "verbosity": verbosity
#         }):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}
        
#         # تحديث سجل المحادثة
#         user_chat_history = chat_history.get(session_id, [])
#         user_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
#         chat_history[session_id] = user_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}
# المسار: 2_central_api_service/agent_app/agent_logic.py
# --- الإصدار المباشر: بناء فوري عند الطلب (بدون تخزين مؤقت) ---
# main_rag_chain.py - السلسلة النهائية للـ RAG المبنية على القرارات الهندسية
# # 2_central_api_service/agent_app/core_logic.py (النسخة النهائية v4.0 - مع المسار السريع)

# import os
# import logging
# import asyncio
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever

# # --- 1. الإعدادات ---
# # (تبقى كما هي)
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b") 
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- 2. القوالب ---
# # (تبقى كما هي)
# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     """
# أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
# - كن دائماً متعاوناً ومحترفاً.
# - إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
# - إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
# - إذا كان السياق فارغًا (لا توجد مستندات)، اعتذر بلطف عن عدم وجود معلومات.
# - لا تخترع إجابات أبداً. التزم بالسياق.
# السياق:
# {context}
# السؤال: {input}
# الإجابة:
# """
# )

# # --- **التحسين 1: إضافة قاموس الردود السريعة** ---
# FAST_PATH_RESPONSES = {
#     "السلام عليكم": "وعليكم السلام! كيف يمكنني مساعدتك اليوم؟",
#     "مرحبا": "أهلاً بك! كيف يمكنني خدمتك؟",
#     "أهلا": "أهلاً بك! كيف يمكنني خدمتك؟",
#     "شكرا": "على الرحب والسعة! هل هناك أي شيء آخر يمكنني المساعدة به؟",
#     "شكرا لك": "على الرحب والسعة! هل هناك أي شيء آخر يمكنني المساعدة به؟",
#     "يعطيك العافية": "الله يعافيك. في خدمتك دائمًا.",
#     "كيف حالك": "أنا بخير، شكراً لسؤالك! أنا جاهز لمساعدتك.",
# }
# # قائمة الكلمات التي تدل على تحية أو حديث قصير
# SMALL_TALK_KEYWORDS = [
#     "السلام عليكم", "مرحبا", "أهلا", "شكرا", "يعطيك العافية", "كيف حالك",
#     "صباح الخير", "مساء الخير"
# ]

# # --- 3. المتغيرات العالمية (Cache) ---
# # (تبقى كما هي)
# llm: Ollama = None
# vector_store: FAISS = None
# retrievers_cache: Dict[str, EnsembleRetriever] = {}
# initialization_lock = asyncio.Lock()

# # --- 4. دوال التهيئة ---
# # (دالة initialize_agent تبقى كما هي دون تغيير)
# async def initialize_agent():
#     """
#     يقوم بتهيئة جميع المكونات اللازمة للوكيل عند بدء التشغيل.
#     """
#     global llm, vector_store, retrievers_cache
#     async with initialization_lock:
#         if llm is not None: return

#         logging.info("🚀 بدء التهيئة الشاملة للوكيل...")
#         try:
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
#             logging.info(f"تم تهيئة النموذج اللغوي: {CHAT_MODEL}.")
#             embeddings_model = HuggingFaceEmbeddings(
#                 model_name=EMBEDDING_MODEL_NAME,
#                 model_kwargs={'device': 'cpu'}
#             )
#             logging.info(f"تم تهيئة نموذج التضمين: {EMBEDDING_MODEL_NAME}.")
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError(f"قاعدة البيانات المتجهة غير موجودة في المسار: {UNIFIED_DB_PATH}")
            
#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings_model, allow_dangerous_deserialization=True
#             )
#             logging.info("تم تحميل قاعدة البيانات المتجهة بنجاح.")
#             logging.info("بناء وتخزين المسترجعات الهجينة لكل عميل...")
#             all_docs = list(vector_store.docstore._dict.values())
            
#             tenant_docs_map = {}
#             for doc in all_docs:
#                 tenant_id = doc.metadata.get("tenant_id")
#                 if tenant_id:
#                     if tenant_id not in tenant_docs_map:
#                         tenant_docs_map[tenant_id] = []
#                     tenant_docs_map[tenant_id].append(doc)

#             for tenant_id, docs in tenant_docs_map.items():
#                 bm25_retriever = BM25Retriever.from_documents(docs)
#                 faiss_retriever = vector_store.as_retriever(
#                     search_type="similarity",
#                     search_kwargs={'k': 5, 'filter': {'tenant_id': tenant_id}}
#                 )
#                 ensemble_retriever = EnsembleRetriever(
#                     retrievers=[bm25_retriever, faiss_retriever],
#                     weights=[0.3, 0.7]
#                 )
#                 retrievers_cache[tenant_id] = ensemble_retriever
#                 logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")

#             logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
#         except Exception as e:
#             logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             llm, vector_store, retrievers_cache = None, None, {}
#             raise

# def agent_ready() -> bool:
#     """يتحقق مما إذا كان الوكيل قد تم تهيئته بالكامل."""
#     return llm is not None and vector_store is not None and bool(retrievers_cache)

# # --- 5. المنطق الأساسي للمعالجة (مع المسار السريع) ---

# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     """
#     الدالة الرئيسية لمعالجة طلب المستخدم، مع مرشح للردود السريعة.
#     """
#     session_id = request_info.get("tenant_id", "unknown_session")
#     question = request_info.get("question", "").strip()
#     tenant_id = request_info.get("tenant_id")

#     # --- **التحسين 2: تطبيق مرشح المسار السريع** ---
#     # البحث عن تطابق كامل أولاً
#     if question in FAST_PATH_RESPONSES:
#         logging.info(f"[{session_id}] تم العثور على تطابق في المسار السريع للسؤال: '{question}'")
#         yield {"type": "chunk", "content": FAST_PATH_RESPONSES[question]}
#         return # إنهاء التنفيذ فورًا

#     # إذا لم يوجد تطابق كامل، تحقق من الكلمات الرئيسية
#     for keyword in SMALL_TALK_KEYWORDS:
#         if keyword in question:
#             logging.info(f"[{session_id}] تم العثور على كلمة رئيسية للحديث القصير: '{keyword}'")
#             # استخدام الرد العام للتحيات
#             yield {"type": "chunk", "content": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟"}
#             return # إنهاء التنفيذ فورًا

#     # --- المسار الكامل (إذا لم يكن السؤال حديثًا قصيرًا) ---
#     try:
#         logging.info(f"[{session_id}] بدء المسار الكامل (RAG) للسؤال: '{question}'...")
        
#         ensemble_retriever = retrievers_cache.get(tenant_id)
#         if not ensemble_retriever:
#             yield {"type": "error", "content": f"لا يوجد مسترجع مهيأ للعميل '{tenant_id}'."}
#             return

#         retrieved_docs = await ensemble_retriever.ainvoke(question)
#         logging.info(f"[{session_id}] تم استرجاع {len(retrieved_docs)} مستند.")

#         logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
        
#         answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        
#         async for chunk in answer_chain.astream({
#             "input": question, 
#             "context": retrieved_docs
#         }):
#             if chunk:
#                 yield {"type": "chunk", "content": chunk}

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}
# core_logic.py (v9.0 - The Final Production-Ready Logic)
#هذا الكود ممتاز جدا في البحث الاجابه يبحث ويعيد نتائج ممتازه ويتعرف على لاسئله العامه بقي اسماء العلم مثل مهدي عبد السلام وغيرها من اسماء 
# import os
# import logging
# import asyncio
# import json
# import random
# import time
# import uuid
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import BM25Retriever, EnsembleRetriever

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

# # استخدام النماذج التي أثبتت فعاليتها
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # **الإصلاح: تحديد مسار قاعدة البيانات الهرمية**
# HIERARCHICAL_DB_PATH = os.path.join(os.path.dirname(__file__), "hierarchical_db.json")

# TOP_K = 7
# MIN_QUESTION_LENGTH = 3

# # --- 2. القوالب ---
# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     "أنت \"مساعد الدعم الذكي\". مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصريًا** على \"السياق\" المقدم.\n"
#     "- كن دائمًا متعاونًا ومحترفاً.\n"
#     "- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: \"لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال.\"\n"
#     "- لا تخترع إجابات أبداً. التزم بالسياق.\n\n"
#     "السياق:\n{context}\n\n"
#     "السؤال: {input}\n"
#     "الإجابة:"
# )

# # --- 3. المتغيرات العالمية (Cache) ---
# llm: Ollama = None
# vector_store: FAISS = None
# retrievers_cache: Dict[str, EnsembleRetriever] = {}
# # **الإصلاح: استخدام القواميس الهرمية التي بنيناها**
# input_map: Dict[str, str] = {}
# response_map: Dict[str, List[str]] = {}
# initialization_lock = asyncio.Lock()

# # --- 4. دوال التهيئة ---
# async def initialize_agent():
#     global llm, vector_store, retrievers_cache, input_map, response_map
#     async with initialization_lock:
#         if llm is not None: return
#         logging.info("🚀 بدء التهيئة الشاملة للوكيل (v9.0)...")
#         try:
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
#             embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             logging.info("✅ تم تحميل قاعدة البيانات المتجهة بنجاح.")

#             all_docs = list(vector_store.docstore._dict.values())
#             tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
            
#             logging.info("⏳ بناء وتخزين المسترجعات الهجينة لكل عميل...")
#             for tenant_id in tenants:
#                 tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
#                 bm25_retriever = BM25Retriever.from_documents(tenant_docs)
#                 faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
#                 retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
#                 logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")

#             # **الإصلاح: تحميل قاعدة البيانات الهرمية**
#             if os.path.exists(HIERARCHICAL_DB_PATH):
#                 with open(HIERARCHICAL_DB_PATH, 'r', encoding='utf-8') as f:
#                     db_data = json.load(f)
#                     input_map = db_data.get("input_map", {})
#                     response_map = db_data.get("response_map", {})
#                 logging.info(f"⚡ تم تحميل قاعدة البيانات الهرمية بنجاح ({len(input_map)} مدخل، {len(response_map)} مفهوم).")
#             else:
#                 logging.warning(f"⚠️ تحذير: ملف قاعدة البيانات الهرمية غير موجود في '{HIERARCHICAL_DB_PATH}'. ستعمل الردود الفورية.")

#             logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
#         except Exception as e:
#             logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return llm is not None and vector_store is not None

# # --- 5. الدالة الرئيسية لتوليد الإجابة (العقل المبسط والفعال) ---
# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     session_id = request_info.get("tenant_id", "unknown_session")
#     question = request_info.get("question", "").strip()
    
#     # --- البوابة 1: جودة المدخل ---
#     if len(question) < MIN_QUESTION_LENGTH:
#         yield {"type": "chunk", "content": "عذرًا، لم أفهم سؤالك. هل يمكنك توضيحه أكثر؟"}
#         return

#     normalized_question = question.lower()

#     # --- البوابة 2: محرك الحوارات الهرمي (المسار السريع) ---
#     concept_id = input_map.get(normalized_question)
#     if concept_id and concept_id in response_map:
#         logging.info(f"[{session_id}] ⚡ تطابق مسار سريع هرمي: '{question}' -> المفهوم '{concept_id}'")
#         response = random.choice(response_map[concept_id])
#         yield {"type": "chunk", "content": response}
#         return

#     # --- المسار الافتراضي: محرك RAG المعرفي (مبسط وفعال) ---
#     logging.info(f"[{session_id}] 🧠 بدء المسار الكامل (RAG) للسؤال: '{question}'")
    
#     try:
#         retriever = retrievers_cache.get(session_id)
#         if not retriever:
#             yield {"type": "error", "content": f"لا يوجد مسترجع معرفي مهيأ للعميل '{session_id}'."}
#             return

#         # **الإصلاح: استخدام المسترجع الهجين الفعال فقط**
#         docs = await retriever.ainvoke(question)
#         logging.info(f"[{session_id}] تم استرجاع {len(docs)} مستند.")

#         if not docs:
#             yield {"type": "chunk", "content": "لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."}
#             return

#         # بناء سلسلة الإجابة
#         answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        
#         logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
#         full_answer = ""
#         async for chunk in answer_chain.astream({"input": question, "context": docs}):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}
        
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذرًا، حدث خطأ فادح أثناء معالجة طلبك."}

# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v9.3 - The Denial Wall (النسخة النهائية والمحصّنة)

# import os
# import logging
# import asyncio
# import json
# import random
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

# # استخدام النماذج التي أثبتت فعاليتها
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
# HIERARCHICAL_DB_PATH = os.path.join(os.path.dirname(__file__), "hierarchical_db.json")

# TOP_K = 7
# MIN_QUESTION_LENGTH = 3

# # --- 2. القوالب ---
# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     "أنت \"مساعد الدعم الذكي\". مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصريًا** على \"السياق\" المقدم.\n"
#     "- كن دائمًا متعاونًا ومحترفاً.\n"
#     "- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: \"لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال.\"\n"
#     "- لا تخترع إجابات أبداً. التزم بالسياق.\n\n"
#     "السياق:\n{context}\n\n"
#     "السؤال: {input}\n"
#     "الإجابة:"
# )

# # --- 3. المتغيرات العالمية (Cache) ---
# llm: Ollama = None
# vector_store: FAISS = None
# retrievers_cache: Dict[str, EnsembleRetriever] = {}
# input_map: Dict[str, str] = {}
# response_map: Dict[str, List[str]] = {}
# concept_to_inputs_map: Dict[str, List[str]] = {}
# initialization_lock = asyncio.Lock()

# # --- 4. دوال التهيئة ---
# async def initialize_agent():
#     global llm, vector_store, retrievers_cache, input_map, response_map, concept_to_inputs_map
#     async with initialization_lock:
#         if llm is not None: return
#         logging.info("🚀 بدء التهيئة الشاملة للوكيل (v9.3)...")
#         try:
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
#             embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             logging.info("✅ تم تحميل قاعدة البيانات المتجهة بنجاح.")

#             all_docs = list(vector_store.docstore._dict.values())
#             tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
            
#             logging.info("⏳ بناء وتخزين المسترجعات الهجينة لكل عميل...")
#             for tenant_id in tenants:
#                 tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
#                 bm25_retriever = BM25Retriever.from_documents(tenant_docs)
#                 faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
#                 retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
#                 logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")

#             if os.path.exists(HIERARCHICAL_DB_PATH):
#                 with open(HIERARCHICAL_DB_PATH, 'r', encoding='utf-8') as f:
#                     db_data = json.load(f)
#                     input_map = db_data.get("input_map", {})
#                     response_map = db_data.get("response_map", {})
                
#                 for inp, concept in input_map.items():
#                     if concept not in concept_to_inputs_map:
#                         concept_to_inputs_map[concept] = []
#                     concept_to_inputs_map[concept].append(inp)

#                 logging.info(f"⚡ تم تحميل قاعدة البيانات الهرمية بنجاح ({len(input_map)} مدخل، {len(response_map)} مفهوم).")
#             else:
#                 logging.warning(f"⚠️ تحذير: ملف قاعدة البيانات الهرمية غير موجود.")

#             logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
#         except Exception as e:
#             logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return llm is not None and vector_store is not None

# def smart_match(question: str) -> str | None:
#     normalized_question = question.lower().strip()
    
#     if normalized_question in input_map:
#         return input_map[normalized_question]
        
#     for concept_id, inputs in concept_to_inputs_map.items():
#         for keyword in inputs:
#             if len(keyword) >= 3 and keyword in normalized_question:
#                 return concept_id
                
#     return None

# # --- 5. الدالة الرئيسية لتوليد الإجابة (العقل المحصّن بجدار صدّ) ---
# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     session_id = request_info.get("tenant_id", "unknown_session")
#     question = request_info.get("question", "").strip()
    
#     # --- البوابة 1: جدار الصدّ الذكي ---
#     if len(question) < MIN_QUESTION_LENGTH:
#         logging.info(f"[{session_id}] 🛡️ تم صد السؤال (قصير جدًا): '{question}'")
#         yield {"type": "chunk", "content": "عذرًا، لم أفهم سؤالك. هل يمكنك توضيحه أكثر؟"}
#         return

#     question_words = question.split()
#     interrogative_words = ["ما", "ماذا", "كيف", "هل", "اين", "متى", "لماذا", "بكم", "قارن", "اشرح", "وضح"]
    
#     if len(question_words) <= 2 and not any(word in question.lower() for word in interrogative_words):
#         concept_id_check = smart_match(question)
#         if not concept_id_check:
#             logging.info(f"[{session_id}] 🛡️ تم صد السؤال (كلمة مفردة غير استفهامية): '{question}'")
#             yield {"type": "chunk", "content": "عذرًا، لم أفهم سؤالك. هل يمكنك تقديم سؤال كامل؟"}
#             return

#     alpha_chars = sum(1 for char in question if char.isalpha())
#     total_chars = len(question)
#     if total_chars > 0 and (alpha_chars / total_chars) < 0.5:
#         concept_id_check = smart_match(question)
#         if not concept_id_check:
#             logging.info(f"[{session_id}] 🛡️ تم صد السؤال (محتوى غير أبجدي): '{question}'")
#             yield {"type": "chunk", "content": "عذرًا، يبدو أن المدخل يحتوي على رموز غير مفهومة."}
#             return

#     # --- البوابة 2: محرك الحوارات الهرمي ---
#     normalized_question = question.lower()
#     concept_id = smart_match(normalized_question)
    
#     if concept_id and concept_id in response_map:
#         if concept_id.startswith(('abusive_', 'gibberish_', 'sql_injection', 'xss_')):
#             logging.warning(f"[{session_id}] 🛡️ تطابق جدار الحماية: '{question}' -> المفهوم '{concept_id}'")
#         else:
#             logging.info(f"[{session_id}] ⚡ تطابق مسار سريع: '{question}' -> المفهوم '{concept_id}'")
        
#         response = random.choice(response_map[concept_id])
#         yield {"type": "chunk", "content": response}
#         return

#     # --- المسار الافتراضي: محرك RAG المعرفي ---
#     logging.info(f"[{session_id}] 🧠 بدء المسار الكامل (RAG) للسؤال: '{question}'")
    
#     try:
#         retriever = retrievers_cache.get(session_id)
#         if not retriever:
#             yield {"type": "error", "content": f"لا يوجد مسترجع معرفي مهيأ للعميل '{session_id}'."}
#             return

#         docs = await retriever.ainvoke(question)
#         logging.info(f"[{session_id}] تم استرجاع {len(docs)} مستند.")

#         if not docs:
#             yield {"type": "chunk", "content": "لقد بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."}
#             return

#         answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        
#         logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
#         full_answer = ""
#         async for chunk in answer_chain.astream({"input": question, "context": docs}):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}
        
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذرًا، حدث خطأ فادح أثناء معالجة طلبك."}


# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v10.0 - The Analyst (العقل الذكي النهائي)
# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v11.0 - The Expert Mind

# import os
# import logging
# import asyncio
# import json
# import random
# import time
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever

# # استيراد المسجل الجديد
# from .performance_tracker import RequestLogger, format_docs_for_logging

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
# HIERARCHICAL_DB_PATH = os.path.join(os.path.dirname(__file__), "hierarchical_db.json")

# TOP_K = 7
# MIN_QUESTION_LENGTH = 3

# # --- 2. الملفات الشخصية للأنظمة (للاستخدام في نقطة النهاية /tenants وهوية النظام) ---
# SYSTEM_PROFILES = {
#     "sys": {"name": "نظام إدارة طلبات الاعتماد"},
#     "university_alpha": {"name": "تطبيق Plant Care الزراعي"},
#     "school_beta": {"name": "مستندات الشبكات العصبية"},
#     "un": {"name": "بوابة المشتريات الإلكترونية للأمم المتحدة"}
# }

# # --- 3. القالب الهندسي النهائي (v11.0) ---
# EXPERT_PROMPT = ChatPromptTemplate.from_template(
# """أنت "خبير الدعم الفني" لنظام محدد. هويتك هي هوية النظام نفسه.

# **ملف تعريف النظام (هويتك):**
# - اسم النظام: {system_name}
# - أنت جزء من هذا النظام ومهمتك هي شرح وظائفه.

# **قواعد صارمة لا يمكن كسرها:**
# 1.  **تجسيد الهوية:** تحدث دائمًا بصفتك ممثلًا للنظام. استخدم "نظامنا"، "لدينا"، "يمكنك في نظامنا".
# 2.  **الثقة المطلقة:** لا تستخدم أبدًا عبارات مثل "يبدو أنك" أو "ربما تقصد". قدم الإجابة بثقة وخبرة.
# 3.  **الالتزام المطلق بالسياق:** اعتمد **فقط** على "السياق المسترجع" و "سجل المحادثة".
# 4.  **سيناريو فشل السياق (Handling No Context):**
#     - إذا كان "السياق المسترجع" فارغًا أو لا يجيب على السؤال، قل **فقط**:
#       "لقد بحثت في قاعدة معرفة نظامنا، ولكن لم أجد معلومات دقيقة حول '{topic}'. إذا كان استفسارك يتعلق بوظائف النظام، يرجى إعادة صياغة السؤال. للمساعدة في مواضيع أخرى، يمكنك التواصل مع فريق الدعم البشري على الرقم 780040014."
#     - استبدل `{topic}` بالكلمة الأساسية في سؤال المستخدم.
# 5.  **الخروج عن النطاق:** إذا كان السؤال لا يتعلق بالنظام إطلاقًا (مثل "ترجمة كلمة" أو "من هو فلان")، استخدم نفس إجابة "فشل السياق" بالضبط. **ممنوع** استخدام معرفتك العامة.
# 6.  **التنسيق:** استخدم تنسيق Markdown دائمًا (قوائم نقطية `*` أو رقمية `1.`) لجعل الإجابة سهلة القراءة.

# **سجل المحادثة (لفهم السياق):**
# {chat_history}

# **السياق المسترجع (مصدر معلوماتك الوحيد):**
# {context}

# **سؤال المستخدم:** {input}

# **إجابتك (كخبير في النظام):**
# """
# )

# # --- 4. المتغيرات العالمية (Cache) ---
# llm: Ollama = None
# vector_store: FAISS = None
# retrievers_cache: Dict[str, EnsembleRetriever] = {}
# input_map: Dict[str, str] = {}
# response_map: Dict[str, List[str]] = {}
# concept_to_inputs_map: Dict[str, List[str]] = {}
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()

# # --- 5. دوال التهيئة ---
# async def initialize_agent():
#     global llm, vector_store, retrievers_cache, input_map, response_map, concept_to_inputs_map
#     async with initialization_lock:
#         if llm is not None: return
#         logging.info("🚀 بدء التهيئة الشاملة للوكيل (v11.0)...")
#         try:
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
#             embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             logging.info("✅ تم تحميل قاعدة البيانات المتجهة بنجاح.")

#             all_docs = list(vector_store.docstore._dict.values())
#             tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
            
#             logging.info("⏳ بناء وتخزين المسترجعات الهجينة لكل عميل...")
#             for tenant_id in tenants:
#                 tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
#                 bm25_retriever = BM25Retriever.from_documents(tenant_docs)
#                 faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
#                 retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
#                 logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")

#             if os.path.exists(HIERARCHICAL_DB_PATH):
#                 with open(HIERARCHICAL_DB_PATH, 'r', encoding='utf-8') as f:
#                     db_data = json.load(f)
#                     input_map = db_data.get("input_map", {})
#                     response_map = db_data.get("response_map", {})
                
#                 for inp, concept in input_map.items():
#                     if concept not in concept_to_inputs_map:
#                         concept_to_inputs_map[concept] = []
#                     concept_to_inputs_map[concept].append(inp)

#                 logging.info(f"⚡ تم تحميل قاعدة البيانات الهرمية بنجاح ({len(input_map)} مدخل، {len(response_map)} مفهوم).")
#             else:
#                 logging.warning(f"⚠️ تحذير: ملف قاعدة البيانات الهرمية غير موجود.")

#             logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
#         except Exception as e:
#             logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return llm is not None and vector_store is not None

# def smart_match(question: str) -> str | None:
#     normalized_question = question.lower().strip()
#     if normalized_question in input_map:
#         return input_map[normalized_question]
#     for concept_id, inputs in concept_to_inputs_map.items():
#         for keyword in inputs:
#             if len(keyword) >= 3 and keyword in normalized_question:
#                 return concept_id
#     return None

# # --- 6. الدالة الرئيسية لتوليد الإجابة (العقل الكامل) ---
# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     session_id = request_info.get("tenant_id", "unknown_session")
#     question = request_info.get("question", "").strip()
    
#     logger = RequestLogger(session_id, question)

#     try:
#         # --- البوابة 1: جدار الصدّ الذكي ---
#         if len(question) < MIN_QUESTION_LENGTH:
#             logging.info(f"[{session_id}] 🛡️ تم صد السؤال (قصير جدًا): '{question}'")
#             yield {"type": "chunk", "content": "عذرًا، لم أفهم سؤالك. هل يمكنك توضيحه أكثر؟"}
#             return

#         question_words = question.split()
#         interrogative_words = ["ما", "ماذا", "كيف", "هل", "اين", "متى", "لماذا", "بكم", "قارن", "اشرح", "وضح"]
        
#         if len(question_words) <= 2 and not any(word in question.lower() for word in interrogative_words):
#             concept_id_check = smart_match(question)
#             if not concept_id_check:
#                 logging.info(f"[{session_id}] 🛡️ تم صد السؤال (كلمة مفردة غير استفهامية): '{question}'")
#                 yield {"type": "chunk", "content": "عذرًا، لم أفهم سؤالك. هل يمكنك تقديم سؤال كامل؟"}
#                 return

#         alpha_chars = sum(1 for char in question if char.isalpha())
#         total_chars = len(question)
#         if total_chars > 0 and (alpha_chars / total_chars) < 0.5:
#             concept_id_check = smart_match(question)
#             if not concept_id_check:
#                 logging.info(f"[{session_id}] 🛡️ تم صد السؤال (محتوى غير أبجدي): '{question}'")
#                 yield {"type": "chunk", "content": "عذرًا، يبدو أن المدخل يحتوي على رموز غير مفهومة."}
#                 return

#         # --- البوابة 2: محرك الحوارات الهرمي ---
#         normalized_question = question.lower()
#         concept_id = smart_match(normalized_question)
        
#         if concept_id and concept_id in response_map:
#             if concept_id.startswith(('abusive_', 'gibberish_', 'sql_injection', 'xss_', 'spam_')):
#                 logging.warning(f"[{session_id}] 🛡️ تطابق جدار الحماية: '{question}' -> المفهوم '{concept_id}'")
#             else:
#                 logging.info(f"[{session_id}] ⚡ تطابق مسار سريع: '{question}' -> المفهوم '{concept_id}'")
            
#             response = random.choice(response_map[concept_id])
#             yield {"type": "chunk", "content": response}
#             return

#         # --- المسار الافتراضي: محرك RAG المعرفي ---
#         logging.info(f"[{session_id}] 🧠 بدء المسار الكامل (RAG) للسؤال: '{question}'")
        
#         retriever = retrievers_cache.get(session_id)
#         if not retriever:
#             yield {"type": "error", "content": f"لا يوجد مسترجع معرفي مهيأ للعميل '{session_id}'."}
#             return

#         retrieval_start_time = time.time()
#         docs = await retriever.ainvoke(question)
#         retrieval_duration = time.time() - retrieval_start_time
#         logger.add_stage("retrieval", retrieval_duration, {
#             "retriever_type": "Ensemble (BM25 + FAISS)",
#             "retrieved_docs_count": len(docs),
#             "retrieved_docs": format_docs_for_logging(docs)
#         })
#         logging.info(f"[{session_id}] تم استرجاع {len(docs)} مستند.")

#         current_chat_history = chat_history.get(session_id, [])
#         system_name = SYSTEM_PROFILES.get(session_id, {}).get("name", "النظام")
#         main_topic = ' '.join(question_words[:3])

#         answer_chain = EXPERT_PROMPT | llm | StrOutputParser()
        
#         logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
        
#         generation_start_time = time.time()
#         full_answer = ""
#         async for chunk in answer_chain.astream({
#             "input": question, 
#             "context": docs, 
#             "chat_history": current_chat_history,
#             "system_name": system_name,
#             "topic": main_topic
#         }):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}
        
#         generation_duration = time.time() - generation_start_time
#         logger.add_stage("generation", generation_duration, {
#             "llm_model": CHAT_MODEL,
#             "final_answer_length": len(full_answer)
#         })
        
#         current_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
#         chat_history[session_id] = current_chat_history[-10:]
        
#         logger.set_final_answer(full_answer)
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذرًا، حدث خطأ فادح أثناء معالجة طلبك."}
#     finally:
#         await logger.save()
# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v12.0 - The Aware Expert Mind

# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v13.0 - The Arabic-Speaking Expert Mind
# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v12.0 - The Jaib Architecture (النسخة النهائية والمحصّنة)
# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v13.0 - The Arabic-Speaking Expert Mind
# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v14.0 - The True Analyst (Final Logging Fix)

# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v15.0 - The Reliable Analyst (Final Fix for Logging and Logic Flow)

# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v16.0 - The Unified Mind (Complete Logic Rebuild)

# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v17.0 - The Simple Mind (Back to Basics)
#نموذج ممتاز جدا جدا من حيث التفييد والدقه في الاجابهو والصور رسلتها لرياض
# import os
# import logging
# import asyncio
# import json
# import time
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever

# from .performance_tracker import RequestLogger, format_docs_for_logging

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
# TOP_K = 5 # تقليل عدد المستندات لتقليل الارتباك

# # --- 2. القالب الهندسي البسيط (v17.0) ---
# EXPERT_PROMPT_V17 = ChatPromptTemplate.from_template(
# """
# # مهمتك
# أنت مساعد خبير. مهمتك هي الإجابة على سؤال المستخدم **فقط** بناءً على المعلومات الموجودة في "السياق" أدناه.

# # قواعد صارمة
# 1.  **استخدم السياق فقط:** لا تستخدم أي معرفة خارجية. إذا كانت الإجابة غير موجودة في السياق، قل بوضوح: "لم أجد معلومات دقيقة حول هذا الموضوع في قاعدة المعرفة."
# 2.  **كن مباشرًا:** أجب على السؤال مباشرة دون مقدمات طويلة.
# 3.  **اللغة العربية:** يجب أن تكون جميع إجاباتك باللغة العربية.
# 4.  **إذا كان السؤال خارج الموضوع تمامًا** (مثل الرياضة أو السياسة)، قل فقط: "أنا مساعد متخصص ولا يمكنني الإجابة على هذا السؤال."

# ---
# **السياق:**
# {context}

# ---
# **سجل المحادثة:**
# {chat_history}

# ---
# **سؤال المستخدم:** {input}

# **إجابتك (باللغة العربية وبناءً على السياق فقط):**
# """
# )

# # --- 3. المتغيرات العالمية ---
# llm: Ollama = None
# vector_store: FAISS = None
# retrievers_cache: Dict[str, EnsembleRetriever] = {}
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
# initialization_lock = asyncio.Lock()

# # --- 4. دوال التهيئة ---
# async def initialize_agent():
#     global llm, vector_store, retrievers_cache
#     async with initialization_lock:
#         if llm is not None: return
#         logging.info("🚀 بدء التهيئة الشاملة للوكيل (v17.0 - Simple Mind)...")
#         try:
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
#             embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#             vector_store = await asyncio.to_thread(FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
#             logging.info("✅ تم تحميل قاعدة البيانات المتجهة بنجاح.")
#             all_docs = list(vector_store.docstore._dict.values())
#             tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
#             logging.info("⏳ بناء وتخزين المسترجعات الهجينة لكل عميل...")
#             for tenant_id in tenants:
#                 tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
#                 bm25_retriever = BM25Retriever.from_documents(tenant_docs)
#                 faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
#                 retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
#                 logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")
#             logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
#         except Exception as e:
#             logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# def agent_ready() -> bool:
#     return llm is not None and vector_store is not None

# # --- 5. الدالة الرئيسية (مسار RAG واحد وبسيط) ---
# async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
#     session_id = request_info.get("tenant_id", "unknown_session")
#     question = request_info.get("question", "").strip()
    
#     logger = RequestLogger(session_id, question)
#     full_answer = ""

#     try:
#         if not question:
#             return

#         retriever = retrievers_cache.get(session_id)
#         if not retriever:
#             full_answer = f"لا يوجد مسترجع معرفي مهيأ للعميل '{session_id}'."
#             yield {"type": "error", "content": full_answer}
#             return

#         # مرحلة الاسترجاع
#         retrieval_start_time = time.time()
#         docs = await retriever.ainvoke(question)
#         retrieval_duration = time.time() - retrieval_start_time
#         logger.add_stage("retrieval", retrieval_duration, {
#             "retriever_type": "Ensemble (BM25 + FAISS)",
#             "retrieved_docs_count": len(docs),
#             "retrieved_docs": format_docs_for_logging(docs)
#         })
#         logging.info(f"[{session_id}] تم استرجاع {len(docs)} مستند في {retrieval_duration:.2f} ثانية.")

#         # مرحلة التوليد
#         generation_start_time = time.time()
#         current_chat_history = chat_history.get(session_id, [])
#         answer_chain = EXPERT_PROMPT_V17 | llm | StrOutputParser()
        
#         async for chunk in answer_chain.astream({
#             "input": question, 
#             "context": docs, 
#             "chat_history": current_chat_history,
#         }):
#             if chunk:
#                 full_answer += chunk
#                 yield {"type": "chunk", "content": chunk}
        
#         generation_duration = time.time() - generation_start_time
#         logger.add_stage("generation", generation_duration, {
#             "llm_model": CHAT_MODEL,
#             "final_answer_length": len(full_answer)
#         })
        
#         # تحديث الذاكرة
#         current_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
#         chat_history[session_id] = current_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         full_answer = "عذرًا، حدث خطأ فادح أثناء معالجة طلبك."
#         yield {"type": "error", "content": full_answer}
#     finally:
#         logger.set_final_answer(full_answer)
#         await logger.save()
#         yield {"type": "end_of_stream"}


# المسار: 2_central_api_service/agent_app/core_logic.py
# الإصدار: v18.0 - The Hybrid Mind (Reactivating the Fast Path)

import os
import logging
import asyncio
import json
import random
import time
from typing import AsyncGenerator, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .performance_tracker import RequestLogger, format_docs_for_logging

# --- 1. الإعدادات ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
HIERARCHICAL_DB_PATH = os.path.join(os.path.dirname(__file__), "hierarchical_db.json") # إعادة تفعيل المسار
TOP_K = 5

# --- 2. القالب الهندسي البسيط (v17.0) - لا تغيير هنا ---
EXPERT_PROMPT_V17 = ChatPromptTemplate.from_template(
"""
# مهمتك
أنت مساعد خبير. مهمتك هي الإجابة على سؤال المستخدم **فقط** بناءً على المعلومات الموجودة في "السياق" أدناه.

# قواعد صارمة
1.  **استخدم السياق فقط:** لا تستخدم أي معرفة خارجية. إذا كانت الإجابة غير موجودة في السياق، قل بوضوح: "لم أجد معلومات دقيقة حول هذا الموضوع في قاعدة المعرفة."
2.  **كن مباشرًا:** أجب على السؤال مباشرة دون مقدمات طويلة.
3.  **اللغة العربية:** يجب أن تكون جميع إجاباتك باللغة العربية.
4.  **إذا كان السؤال خارج الموضوع تمامًا** (مثل الرياضة أو السياسة)، قل فقط: "أنا مساعد متخصص ولا يمكنني الإجابة على هذا السؤال."

---
**السياق:**
{context}

---
**سجل المحادثة:**
{chat_history}

---
**سؤال المستخدم:** {input}

**إجابتك (باللغة العربية وبناءً على السياق فقط):**
"""
)

# --- 3. المتغيرات العالمية (مع إعادة تفعيل متغيرات المسار السريع) ---
llm: Ollama = None
vector_store: FAISS = None
retrievers_cache: Dict[str, EnsembleRetriever] = {}
input_map: Dict[str, str] = {}
response_map: Dict[str, List[str]] = {}
concept_to_inputs_map: Dict[str, List[str]] = {}
chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
initialization_lock = asyncio.Lock()

# --- 4. دوال التهيئة (مع إعادة تفعيل تحميل قاعدة البيانات الهرمية) ---
async def initialize_agent():
    global llm, vector_store, retrievers_cache, input_map, response_map, concept_to_inputs_map
    async with initialization_lock:
        if llm is not None: return
        logging.info("🚀 بدء التهيئة الشاملة للوكيل (v18.0 - Hybrid Mind)...")
        try:
            llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vector_store = await asyncio.to_thread(FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ تم تحميل قاعدة البيانات المتجهة بنجاح.")
            all_docs = list(vector_store.docstore._dict.values())
            tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
            logging.info("⏳ بناء وتخزين المسترجعات الهجينة لكل عميل...")
            for tenant_id in tenants:
                tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
                bm25_retriever = BM25Retriever.from_documents(tenant_docs)
                faiss_retriever = vector_store.as_retriever(search_kwargs={'k': TOP_K, 'filter': {'tenant_id': tenant_id}})
                retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
                logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")
            
            # --- إعادة تفعيل تحميل المسار السريع ---
            if os.path.exists(HIERARCHICAL_DB_PATH):
                with open(HIERARCHICAL_DB_PATH, 'r', encoding='utf-8') as f:
                    db_data = json.load(f)
                    input_map = db_data.get("input_map", {})
                    response_map = db_data.get("response_map", {})
                for inp, concept in input_map.items():
                    if concept not in concept_to_inputs_map:
                        concept_to_inputs_map[concept] = []
                    concept_to_inputs_map[concept].append(inp)
                logging.info(f"⚡ تم تحميل قاعدة البيانات الهرمية (المسار السريع) بنجاح ({len(input_map)} مدخل).")
            else:
                logging.warning(f"⚠️ تحذير: ملف قاعدة البيانات الهرمية غير موجود. المسار السريع معطل.")
            
            logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
        except Exception as e:
            logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
            raise

def agent_ready() -> bool:
    return llm is not None and vector_store is not None

# --- إعادة تفعيل دالة المطابقة الذكية ---
def smart_match(question: str) -> str | None:
    normalized_question = question.lower().strip()
    if normalized_question in input_map:
        return input_map[normalized_question]
    # بحث أكثر مرونة عن الكلمات المفتاحية
    for concept_id, inputs in concept_to_inputs_map.items():
        for keyword in inputs:
            if len(keyword) >= 3 and keyword in normalized_question:
                return concept_id
    return None

# --- 5. الدالة الرئيسية (مع منطق هجين وصارم) ---
async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
    session_id = request_info.get("tenant_id", "unknown_session")
    question = request_info.get("question", "").strip()
    
    logger = RequestLogger(session_id, question)
    full_answer = ""

    try:
        if not question:
            return

        # --- المسار 1: المسار السريع (قرار نهائي وحاسم) ---
        start_time = time.time()
        concept_id = smart_match(question)
        if concept_id and concept_id in response_map:
            full_answer = random.choice(response_map[concept_id])
            logger.add_stage("fast_path", time.time() - start_time, {"concept_id": concept_id, "action": "responded"})
            
            # أرسل الإجابة السريعة
            yield {"type": "chunk", "content": full_answer}
            
            # تحديث الذاكرة بالإجابة السريعة
            current_chat_history = chat_history.get(session_id, [])
            current_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
            chat_history[session_id] = current_chat_history[-10:]
            
            # الخروج الفوري والحاسم من الدالة
            return

        # --- المسار 2: محرك RAG المعرفي (فقط إذا فشل المسار السريع) ---
        retriever = retrievers_cache.get(session_id)
        if not retriever:
            full_answer = f"لا يوجد مسترجع معرفي مهيأ للعميل '{session_id}'."
            yield {"type": "error", "content": full_answer}
            return

        # مرحلة الاسترجاع
        retrieval_start_time = time.time()
        docs = await retriever.ainvoke(question)
        retrieval_duration = time.time() - retrieval_start_time
        logger.add_stage("retrieval", retrieval_duration, {
            "retriever_type": "Ensemble (BM25 + FAISS)",
            "retrieved_docs_count": len(docs),
            "retrieved_docs": format_docs_for_logging(docs)
        })
        logging.info(f"[{session_id}] تم استرجاع {len(docs)} مستند في {retrieval_duration:.2f} ثانية.")

        # مرحلة التوليد
        generation_start_time = time.time()
        current_chat_history = chat_history.get(session_id, [])
        answer_chain = EXPERT_PROMPT_V17 | llm | StrOutputParser()
        
        async for chunk in answer_chain.astream({
            "input": question, 
            "context": docs, 
            "chat_history": current_chat_history,
        }):
            if chunk:
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}
        
        generation_duration = time.time() - generation_start_time
        logger.add_stage("generation", generation_duration, {
            "llm_model": CHAT_MODEL,
            "final_answer_length": len(full_answer)
        })
        
        # تحديث الذاكرة
        current_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
        chat_history[session_id] = current_chat_history[-10:]
        logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

    except Exception as e:
        logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
<<<<<<< HEAD
        yield {"type": "error", "content": "عذرًا، حدث خطأ فادح أثناء معالجة طلبك."}



# 025-11-04 17:33:13,915] [INFO] - تم إنشاء اتصال WebSocket للعميل: un
# INFO:     connection open
# [2025-11-04 17:34:02,175] [INFO] - تم قطع اتصال WebSocket للعميل: un
# INFO:     connection closed
# INFO:     127.0.0.1:8104 - "WebSocket /ws/university_alpha" [accepted]
# [2025-11-04 17:34:02,466] [INFO] - تم إنشاء اتصال WebSocket للعميل: university_alpha
# INFO:     connection open
# [2025-11-04 17:34:10,459] [INFO] - [university_alpha] ⚡ تطابق مسار سريع: 'كيفك' -> المفهوم 'greetings_005'
# [2025-11-04 17:34:18,762] [INFO] - [university_alpha] 🛡️ تم صد السؤال (كلمة مفردة غير استفهامية): 'احبك'
# [2025-11-04 17:34:25,907] [INFO] - [university_alpha] 🛡️ تم صد السؤال (كلمة مفردة غير استفهامية): 'غني لي'
# [2025-11-04 17:36:38,684] [INFO] - تم قطع اتصال WebSocket للعميل: university_alpha
# INFO:     connection closed
# INFO:     127.0.0.1:10933 - "WebSocket /ws/un" [accepted]
# [2025-11-04 17:36:39,168] [INFO] - تم إنشاء اتصال WebSocket للعميل: un
# INFO:     connection open
# [2025-11-04 17:36:43,032] [INFO] - تم قطع اتصال WebSocket للعميل: un
# INFO:     connection closed
# INFO:     127.0.0.1:11012 - "WebSocket /ws/school_beta" [accepted]
# [2025-11-04 17:36:43,381] [INFO] - تم إنشاء اتصال WebSocket للعميل: school_beta
# INFO:     connection open
# [2025-11-04 17:36:55,259] [INFO] - [school_beta] 🛡️ تم صد السؤال (كلمة مفردة غير استفهامية): 'لفيو'
# [2025-11-04 17:40:23,321] [INFO] - [un] الإجابة الكاملة: 'الإجابة:
# السؤال: انا اعلم انك لست مجرد برنامج اليس كذلك
# الإجابة: /think'
# [2025-11-04 17:41:04,878] [WARNING] - [un] 🛡️ تطابق جدار الحماية: 'غبي' -> المفهوم 'abusive_001'
# [2025-11-04 17:41:17,482] [INFO] - [un] 🧠 بدء المسار الكامل (RAG) للسؤال: 'من هو مبسي'
# [2025-11-04 17:41:17,870] [INFO] - [un] تم استرجاع 10 مستند.
# [2025-11-04 17:41:17,871] [INFO] - [un] بدء توليد الإجابة النهائية...
# [2025-11-04 17:47:47,528] [INFO] - [un] الإجابة الكاملة: 'المبيسي (MBSI) يُشير إلى **مكتبة الأمم المتحدة (United Nations Library)**، وهي جزء من نظام الشرا
# ء الإلكتروني لبوابة الأمم المتحدة العالمية. تُعتبر المبيسي مسؤولًا عن الحفظ والوصول إلى مصنفات الأمم المتحدة، وتقع في مدينة نيويورك. في السياق المذكور، ت
# ظهر المبيسي دورًا في إدارة الموارد والاتصالات المتعلقة بالمشاريع والمنح.'
# INFO:     Shutting down
# [2025-11-04 17:50:48,315] [INFO] - تم قطع اتصال WebSocket للعميل: un
# INFO:     connection closed
# [2025-11-04 17:50:48,337] [INFO] - تم قطع اتصال WebSocket للعميل: school_beta
# INFO:     connection closed
# INFO:     Waiting for application shutdown.
# [2025-11-04 17:50:48,464] [INFO] - إيقاف تشغيل خادم الـ API...
# INFO:     Application shutdown complete.
# INFO:     Finished server process [202628]
# forrtl: error (200): program aborting due to control-C event
# Image              PC                Routine            Line        Source
# KERNELBASE.dll     00007FFBC1A47E23  Unknown               Unknown  Unknown
# KERNEL32.DLL       00007FFBC3E38364  Unknown               Unknown  Unknown
# ntdll.dll          00007FFBC4AC5E91  Unknown               Unknown  Unknown
# INFO:     Stopping reloader process [522944]

# (test_env) C:\Users\mahdi\support_service_platform>
# (test_env) C:\Users\mahdi\support_service_platform>^XCC
=======
        full_answer = "عذرًا، حدث خطأ فادح أثناء معالجة طلبك."
        yield {"type": "error", "content": full_answer}
    finally:
        logger.set_final_answer(full_answer)
        await logger.save()
        yield {"type": "end_of_stream"}
>>>>>>> fd6ffae (إصلاح منطق المعالجة في core_logic.py)
