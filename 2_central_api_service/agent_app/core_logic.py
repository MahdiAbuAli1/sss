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
# --- النسخة 3.0: مع محرك إعادة صياغة متقدم (تفكير + أمثلة + قواعد صارمة) ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- النسخة 4.0: مع محرك إعادة صياغة ذكي ومتوازن (أمثلة متعددة + قواعد مرنة) ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- النسخة 4.1: إصلاح خطأ NameError وإعادة المتغيرات العالمية ---

# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 5.0: مع محرك إعادة صياغة ذكي ومتوازن (سلسلة التفكير المنطقية) ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.0: مع Reranker لتحقيق أقصى دقة (الحل النهائي لمشكلة السياق) ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.1: إصلاح مسار استيراد FlashrankRerank ---

# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.2: إصلاح نهائي لمسار استيراد FlashRankRerank ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.3: تطبيق الحل الصحيح باستخدام مكتبة flashrank مباشرة ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.4: استخدام الاسم الصحيح 'Ranker' بناءً على بيئة المستخدم ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.5: إصلاح نهائي لـ TypeError في Ranker ---

# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.6: إصلاح نهائي لـ KeyError في rewriter_chain ---

# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.7: إصلاح نهائي لـ TypeError في reranker.rerank ---

# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.8: إصلاح نهائي لـ TypeError باستخدام الوسائط الموضعية ---

# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 6.9: الإصلاح الحاسم لـ TypeError في reranker.rerank ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 7.0: الإصلاح الجذري والأخير لـ rerank باستخدام RerankRequest ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 8.0: النسخة النهائية مع القالب الأكثر ذكاءً ---
# المسار: 2_central_api_service/agent_app/core_logic.py
# --- الإصدار 10.0: الإصدار النهائي مع إصلاح chat_history ---

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

import os
import logging
import asyncio
import httpx
from typing import AsyncGenerator, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from flashrank import Ranker, RerankRequest

from .performance_tracker import PerformanceLogger

# --- 1. الإعدادات ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# --- 2. الملفات الشخصية للأنظمة ---
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

# --- 3. القالب النهائي (الإصدار 12.0: استخراج الكلمات المفتاحية) ---
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
**أمثلة:**

سؤال المستخدم: ماهو هذا النظام باختصار
الاستعلام المحسّن: نظام إدارة طلبات الاعتماد

سؤال المستخدم: كيفيه الوصول للنظام
الاستعلام المحسّن: كيفية تسجيل الدخول

سؤال المستخدم: ماهي الشبكات العصبيه
الاستعلام المحسّن: الشبكات العصبية

سؤال المستخدم: من هي جورجينا
الاستعلام المحسّن: من هي جورجينا
---

**المهمة المطلوبة:**

سؤال المستخدم: {question}

الاستعلام المحسّن:
"""

# --- باقي القوالب ---
ANSWER_PROMPT = ChatPromptTemplate.from_template("أنت \"مرشد الدعم\"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على \"السياق\" المقدم.\n- كن دائماً متعاوناً ومحترفاً.\n- إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.\n- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: \"بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال.\"\n- لا تخترع إجابات أبداً. التزم بالسياق.\n\nالسياق:\n{context}\n\nالسؤال: {input}\nالإجابة:")

# --- 4. المتغيرات العالمية ---
llm: Ollama = None
vector_store: FAISS = None
reranker: Ranker = None
chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
initialization_lock = asyncio.Lock()
perf_logger = PerformanceLogger()

# --- 5. الدوال الأساسية (معظمها يبقى كما هو) ---

def _load_all_docs_from_faiss(vs: FAISS) -> List[Document]:
    return list(vs.docstore._dict.values())

def _clean_rewritten_query(raw_query: str) -> str:
    lines = raw_query.strip().split('\n')
    for line in reversed(lines):
        cleaned_line = line.strip()
        if cleaned_line:
            if cleaned_line.startswith("الاستعلام المحسّن:"):
                return cleaned_line.replace("الاستعلام المحسّن:", "").strip()
            return cleaned_line
    return raw_query.strip()

async def initialize_agent():
    global llm, vector_store, reranker
    async with initialization_lock:
        if vector_store is not None: return
        logging.info("بدء تهيئة النماذج وقاعدة البيانات و Reranker...")
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=10.0)
            
            llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
            if not os.path.isdir(UNIFIED_DB_PATH):
                raise FileNotFoundError("قاعدة البيانات الموحدة غير موجودة.")

            vector_store = await asyncio.to_thread(
                FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            
            reranker = Ranker()
            
            logging.info("✅ الوكيل جاهز للعمل (مع Reranker).")
        except Exception as e:
            logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
            raise

def agent_ready() -> bool:
    return vector_store is not None and llm is not None and reranker is not None

async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
    question = request_info.get("question", "")
    tenant_id = request_info.get("tenant_id", "default_session")
    k_results = request_info.get("k_results", 10)
    session_id = tenant_id or "default_session"

    if not agent_ready():
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
        return

    user_chat_history = chat_history.get(session_id, [])

    try:
        effective_question = question
        profile = SYSTEM_PROFILES.get(tenant_id)
        
        if profile:
            logging.info(f"[{session_id}] استخدام ملف شخصي متقدم لإعادة صياغة السؤال...")
            rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
            rewriter_chain = rewrite_prompt | llm | StrOutputParser()
            
            raw_rewritten_query = await rewriter_chain.ainvoke({
                "system_name": profile.get("name", ""),
                "system_description": profile.get("description", ""),
                "system_keywords": ", ".join(profile.get("keywords", [])),
                "question": question
            })
            
            effective_question = _clean_rewritten_query(raw_rewritten_query)
            logging.info(f"[{session_id}] السؤال الأصلي: '{question}' -> الاستعلام المحسّن: '{effective_question}'")

        all_docs = _load_all_docs_from_faiss(vector_store)
        tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]

        if not tenant_docs:
            yield {"type": "error", "content": f"لا توجد بيانات للعميل '{tenant_id}'."}
            return

        bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=k_results)
        faiss_retriever = vector_store.as_retriever(
            search_kwargs={'k': k_results, 'filter': {'tenant_id': tenant_id}}
        )
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
        logging.info(f"[{session_id}] بدء الاسترجاع الأولي لـ '{effective_question}'...")
        initial_docs = await ensemble_retriever.ainvoke(effective_question)
        logging.info(f"[{session_id}] تم استرجاع {len(initial_docs)} مستند أولي.")

        logging.info(f"[{session_id}] بدء إعادة الترتيب والفلترة...")
        
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(initial_docs)]
        
        rerank_request = RerankRequest(query=question, passages=passages)
        all_reranked_results = reranker.rerank(rerank_request)
        top_4_results = all_reranked_results[:4]
        
        original_docs_map = {doc.page_content: doc for doc in initial_docs}
        reranked_docs = [original_docs_map[res["text"]] for res in top_4_results if res["text"] in original_docs_map]
        
        logging.info(f"[{session_id}] تم فلترة المستندات إلى {len(reranked_docs)} مستند عالي الصلة.")

        document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        
        logging.info(f"[{session_id}] بدء توليد الإجابة النهائية...")
        full_answer = ""
        
        async for chunk in document_chain.astream({"input": question, "context": reranked_docs, "chat_history": user_chat_history}):
            if chunk:
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}

        user_chat_history.extend([HumanMessage(content=question), AIMessage(content=full_answer)])
        chat_history[session_id] = user_chat_history[-10:]
        logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")

    except Exception as e:
        logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
