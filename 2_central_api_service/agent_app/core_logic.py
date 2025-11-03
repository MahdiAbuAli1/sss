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
# 2_central_api_service/agent_app/core_logic.py (النسخة النهائية v4.0 - مع المسار السريع)

import os
import logging
import asyncio
from typing import AsyncGenerator, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- 1. الإعدادات ---
# (تبقى كما هي)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b") 
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# --- 2. القوالب ---
# (تبقى كما هي)
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
- كن دائماً متعاوناً ومحترفاً.
- إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
- إذا كان السياق فارغًا (لا توجد مستندات)، اعتذر بلطف عن عدم وجود معلومات.
- لا تخترع إجابات أبداً. التزم بالسياق.
السياق:
{context}
السؤال: {input}
الإجابة:
"""
)

# --- **التحسين 1: إضافة قاموس الردود السريعة** ---
FAST_PATH_RESPONSES = {
    "السلام عليكم": "وعليكم السلام! كيف يمكنني مساعدتك اليوم؟",
    "مرحبا": "أهلاً بك! كيف يمكنني خدمتك؟",
    "أهلا": "أهلاً بك! كيف يمكنني خدمتك؟",
    "شكرا": "على الرحب والسعة! هل هناك أي شيء آخر يمكنني المساعدة به؟",
    "شكرا لك": "على الرحب والسعة! هل هناك أي شيء آخر يمكنني المساعدة به؟",
    "يعطيك العافية": "الله يعافيك. في خدمتك دائمًا.",
    "كيف حالك": "أنا بخير، شكراً لسؤالك! أنا جاهز لمساعدتك.",
}
# قائمة الكلمات التي تدل على تحية أو حديث قصير
SMALL_TALK_KEYWORDS = [
    "السلام عليكم", "مرحبا", "أهلا", "شكرا", "يعطيك العافية", "كيف حالك",
    "صباح الخير", "مساء الخير"
]

# --- 3. المتغيرات العالمية (Cache) ---
# (تبقى كما هي)
llm: Ollama = None
vector_store: FAISS = None
retrievers_cache: Dict[str, EnsembleRetriever] = {}
initialization_lock = asyncio.Lock()

# --- 4. دوال التهيئة ---
# (دالة initialize_agent تبقى كما هي دون تغيير)
async def initialize_agent():
    """
    يقوم بتهيئة جميع المكونات اللازمة للوكيل عند بدء التشغيل.
    """
    global llm, vector_store, retrievers_cache
    async with initialization_lock:
        if llm is not None: return

        logging.info("🚀 بدء التهيئة الشاملة للوكيل...")
        try:
            llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
            logging.info(f"تم تهيئة النموذج اللغوي: {CHAT_MODEL}.")
            embeddings_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
            logging.info(f"تم تهيئة نموذج التضمين: {EMBEDDING_MODEL_NAME}.")
            if not os.path.isdir(UNIFIED_DB_PATH):
                raise FileNotFoundError(f"قاعدة البيانات المتجهة غير موجودة في المسار: {UNIFIED_DB_PATH}")
            
            vector_store = await asyncio.to_thread(
                FAISS.load_local, UNIFIED_DB_PATH, embeddings_model, allow_dangerous_deserialization=True
            )
            logging.info("تم تحميل قاعدة البيانات المتجهة بنجاح.")
            logging.info("بناء وتخزين المسترجعات الهجينة لكل عميل...")
            all_docs = list(vector_store.docstore._dict.values())
            
            tenant_docs_map = {}
            for doc in all_docs:
                tenant_id = doc.metadata.get("tenant_id")
                if tenant_id:
                    if tenant_id not in tenant_docs_map:
                        tenant_docs_map[tenant_id] = []
                    tenant_docs_map[tenant_id].append(doc)

            for tenant_id, docs in tenant_docs_map.items():
                bm25_retriever = BM25Retriever.from_documents(docs)
                faiss_retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 5, 'filter': {'tenant_id': tenant_id}}
                )
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever],
                    weights=[0.3, 0.7]
                )
                retrievers_cache[tenant_id] = ensemble_retriever
                logging.info(f"  -> تم بناء المسترجع للعميل: {tenant_id}")

            logging.info("✅ الوكيل جاهز للعمل بكامل طاقته.")
        except Exception as e:
            logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
            llm, vector_store, retrievers_cache = None, None, {}
            raise

def agent_ready() -> bool:
    """يتحقق مما إذا كان الوكيل قد تم تهيئته بالكامل."""
    return llm is not None and vector_store is not None and bool(retrievers_cache)

# --- 5. المنطق الأساسي للمعالجة (مع المسار السريع) ---

async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
    """
    الدالة الرئيسية لمعالجة طلب المستخدم، مع مرشح للردود السريعة.
    """
    session_id = request_info.get("tenant_id", "unknown_session")
    question = request_info.get("question", "").strip()
    tenant_id = request_info.get("tenant_id")

    # --- **التحسين 2: تطبيق مرشح المسار السريع** ---
    # البحث عن تطابق كامل أولاً
    if question in FAST_PATH_RESPONSES:
        logging.info(f"[{session_id}] تم العثور على تطابق في المسار السريع للسؤال: '{question}'")
        yield {"type": "chunk", "content": FAST_PATH_RESPONSES[question]}
        return # إنهاء التنفيذ فورًا

    # إذا لم يوجد تطابق كامل، تحقق من الكلمات الرئيسية
    for keyword in SMALL_TALK_KEYWORDS:
        if keyword in question:
            logging.info(f"[{session_id}] تم العثور على كلمة رئيسية للحديث القصير: '{keyword}'")
            # استخدام الرد العام للتحيات
            yield {"type": "chunk", "content": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟"}
            return # إنهاء التنفيذ فورًا

    # --- المسار الكامل (إذا لم يكن السؤال حديثًا قصيرًا) ---
    try:
        logging.info(f"[{session_id}] بدء المسار الكامل (RAG) للسؤال: '{question}'...")
        
        ensemble_retriever = retrievers_cache.get(tenant_id)
        if not ensemble_retriever:
            yield {"type": "error", "content": f"لا يوجد مسترجع مهيأ للعميل '{tenant_id}'."}
            return

        retrieved_docs = await ensemble_retriever.ainvoke(question)
        logging.info(f"[{session_id}] تم استرجاع {len(retrieved_docs)} مستند.")

        logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
        
        answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        
        async for chunk in answer_chain.astream({
            "input": question, 
            "context": retrieved_docs
        }):
            if chunk:
                yield {"type": "chunk", "content": chunk}

    except Exception as e:
        logging.error(f"[{session_id}] فشل في سلسلة RAG: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}
