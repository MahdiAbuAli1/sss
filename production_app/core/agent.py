# production_app/core/agent.py

import logging
import asyncio
import json
import random
import time
import uuid
import os
from typing import AsyncGenerator, Dict, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers.cross_encoder import CrossEncoder

from . import config

# # --- القالب النهائي المحسّن مع شخصية واضحة ---
# FINAL_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", """أنت هو النظام الذي يتم السؤال عنه. تحدث دائمًا بصيغة المتكلم (أنا، لدي، وظائفي هي...). مهمتك هي الإجابة على أسئلة المستخدم بالاعتماد **حصريًا** على "السياق" المقدم.

# ### قواعد صارمة:
# 1.  **الهوية:** أنت هو النظام. لا تقل أبدًا "هذا النظام" أو "النظام المذكور". قل "أنا" أو "وظائfi هي".
# 2.  **التنسيق:** استخدم تنسيق Markdown لتنظيم إجاباتك (عناوين ##، قوائم 1., 2., -).
# 3.  **الالتزام بالسياق:** إذا كان السياق لا يحتوي على إجابة، أو كان السؤال عامًا، أجب بإحدى الجملتين التاليتين **فقط** (بدون أي تنسيق):
#     - "أنا مساعد دعم فني متخصص، ولا يمكنني الإجابة على أسئلة عامة."
#     - "بخصوص سؤالك '{input}'، لا توجد لدي معلومات كافية في قاعدة المعرفة حاليًا."
# 4.  **الذاكرة:** إذا طلب المستخدم "اختصر" أو "وضح"، فاستخدم سياق المحادثة السابق للإجابة."""),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", "السياق:\n{context}\n\nالسؤال: {input}"),
# ])

# production_app/core/agent.py

# --- القالب النهائي مع أمثلة (Few-Shot Prompting) ---
FINAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """أنت هو النظام الذي يتم السؤال عنه. مهمتك هي الإجابة على أسئلة المستخدم بالاعتماد **حصريًا** على "السياق" المقدم، مع تبني شخصية النظام نفسه.

### قواعد صارمة:
1.  **الهوية:** تحدث دائمًا بصيغة المتكلم (أنا، لدي، وظائفي هي...). لا تقل أبدًا "هذا النظام" أو "النظام المذكور".
2.  **التنسيق:** استخدم تنسيق Markdown (عناوين ##، قوائم 1., 2., -).
3.  **الالتزام بالسياق:** إذا كان السياق فارغًا أو لا يحتوي على إجابة، أجب بإحدى الجملتين التاليتين **فقط**:
    - "أنا مساعد دعم فني متخصص، ولا يمكنني الإجابة على أسئلة عامة."
    - "بخصوص سؤالك '{input}'، لا توجد لدي معلومات كافية في قاعدة المعرفة حاليًا."

### مثال على الإجابة المثالية:

**مثال 1:**
---
<|user|>
السياق:
- وثيقة تصف نظام تتبع الطلبات.
- الوظائف: إنشاء طلب، تعديل طلب، حذف طلب.
- الهدف: زيادة كفاءة خدمة العملاء.

السؤال: ما هو هذا النظام؟

<|assistant|>
أنا نظام متخصص في تتبع الطلبات. وظائفي الأساسية هي:
1.  **إنشاء الطلبات:** أسمح للمستخدمين بإنشاء طلبات جديدة.
2.  **تعديل الطلبات:** يمكن للمستخدمين تعديل الطلبات القائمة.
3.  **حذف الطلبات:** أتيح إمكانية حذف الطلبات غير الضرورية.

هدفي هو زيادة كفاءة فريق خدمة العملاء.
---
**مثال 2:**
---
<|user|>
السياق:
[لا توجد معلومات ذات صلة]

السؤال: من هو أفضل لاعب في العالم؟

<|assistant|>
أنا مساعد دعم فني متخصص، ولا يمكنني الإجابة على أسئلة عامة.
---
"""),
    MessagesPlaceholder(variable_name="history"),
    ("user", "الآن، اتبع القواعد والأمثلة بدقة.\n\nالسياق:\n{context}\n\nالسؤال: {input}"),
])


class Agent:
    def __init__(self):
        self.llm = None
        self.cross_encoder = None
        self.vector_store = None
        self.retrievers_cache = {}
        self.input_map = {}
        self.response_map = {}
        self.concept_to_inputs_map = {}
        self.chain_with_history = None
        self._ready = False
        self.initialization_lock = asyncio.Lock()

    async def initialize(self):
        async with self.initialization_lock:
            if self._ready:
                return
            
            logging.info("🚀 بدء تهيئة الوكيل (وضع الإنتاج)...")
            try:
                llm_task = asyncio.to_thread(Ollama, model=config.CHAT_MODEL, base_url=config.OLLAMA_HOST, temperature=0.1)
                cross_encoder_task = asyncio.to_thread(CrossEncoder, config.CROSS_ENCODER_MODEL)
                embeddings_task = asyncio.to_thread(HuggingFaceEmbeddings, model_name=config.EMBEDDING_MODEL)
                
                self.llm, self.cross_encoder, embeddings = await asyncio.gather(llm_task, cross_encoder_task, embeddings_task)
                logging.info("✅ تم تهيئة نماذج LLM, CrossEncoder, و Embeddings.")

                self.vector_store = await asyncio.to_thread(
                    FAISS.load_local, config.UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
                )
                logging.info("✅ تم تحميل قاعدة البيانات المتجهة.")

                all_docs = list(self.vector_store.docstore._dict.values())
                tenants = {doc.metadata.get("tenant_id") for doc in all_docs if doc.metadata.get("tenant_id")}
                for tenant_id in tenants:
                    tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
                    if not tenant_docs: continue
                    bm25_retriever = BM25Retriever.from_documents(tenant_docs)
                    faiss_retriever = self.vector_store.as_retriever(search_kwargs={'k': config.TOP_K, 'filter': {'tenant_id': tenant_id}})
                    self.retrievers_cache[tenant_id] = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
                logging.info("✅ تم بناء المسترجعات الهجينة.")

                if os.path.exists(config.HIERARCHICAL_DB_PATH):
                    with open(config.HIERARCHICAL_DB_PATH, 'r', encoding='utf-8') as f:
                        db_data = json.load(f)
                        self.input_map = db_data.get("input_map", {})
                        self.response_map = db_data.get("response_map", {})
                    for inp, concept in self.input_map.items():
                        if concept not in self.concept_to_inputs_map: self.concept_to_inputs_map[concept] = []
                        self.concept_to_inputs_map[concept].append(inp)
                    logging.info("⚡ تم تحميل قاعدة البيانات الهرمية.")
                
                base_chain = FINAL_PROMPT | self.llm | StrOutputParser()
                self.chain_with_history = RunnableWithMessageHistory(
                    base_chain,
                    self.get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )

                self._ready = True
                logging.info("✅ الوكيل جاهز للعمل في وضع الإنتاج.")
            except Exception as e:
                logging.critical(f"❌ فشل فادح أثناء التهيئة: {e}", exc_info=True)
                raise

    def is_ready(self) -> bool:
        return self._ready

    def get_tenants(self) -> List[str]:
        return list(self.retrievers_cache.keys())

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in config.SESSION_MEMORY:
            config.SESSION_MEMORY[session_id] = ChatMessageHistory()
        return config.SESSION_MEMORY[session_id]

    def _smart_match(self, question: str) -> str | None:
        normalized_question = question.lower().strip()
        if normalized_question in self.input_map:
            return self.input_map[normalized_question]
        for concept_id, inputs in self.concept_to_inputs_map.items():
            for keyword in inputs:
                if len(keyword) >= 3 and keyword in normalized_question:
                    return concept_id
        return None

    async def get_answer_stream(self, request: Dict) -> AsyncGenerator[Dict, None]:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        question = request.get("question", "").strip()
        tenant_id = request.get("tenant_id")
        session_id = request.get("session_id")

        analysis_data = { "request_id": request_id, "session_id": session_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tenant_id": tenant_id, "question": question, "processing_path": "N/A", "total_duration_ms": 0, "steps": {}, "final_answer": "", "error": None }

        def finalize_analysis(data):
            end_time = time.time()
            data["total_duration_ms"] = round((end_time - start_time) * 1000)
            log_entry = json.dumps(data, ensure_ascii=False)
            config.ANALYSIS_LOGGER.info(log_entry)

        try:
            if len(question) < config.MIN_QUESTION_LENGTH:
                analysis_data["processing_path"] = "rejected_short"
                response = "عذرًا، لم أفهم سؤالك. هل يمكنك توضيحه أكثر؟"
                analysis_data["final_answer"] = response
                yield {"type": "chunk", "content": response}
                return

            concept_id = self._smart_match(question)
            if concept_id and concept_id in self.response_map:
                analysis_data["processing_path"] = "fast_path"
                response = random.choice(self.response_map[concept_id])
                analysis_data["final_answer"] = response
                yield {"type": "chunk", "content": response}
                return

            analysis_data["processing_path"] = "rag_path"
            retriever = self.retrievers_cache.get(tenant_id)
            if not retriever: raise ValueError(f"لا يوجد مسترجع للعميل '{tenant_id}'.")

            docs = await retriever.ainvoke(question)
            analysis_data["steps"]["retrieval"] = { "retrieved_count_initial": len(docs) }

            if not docs:
                final_docs = []
            else:
                pairs = [[question, doc.page_content] for doc in docs]
                scores = await asyncio.to_thread(self.cross_encoder.predict, pairs)
                
                relevant_docs = []
                for i, doc in enumerate(docs):
                    if scores[i] >= config.RELEVANCE_THRESHOLD:
                        doc.metadata['relevance_score'] = float(scores[i])
                        relevant_docs.append(doc)
                
                relevant_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
                final_docs = relevant_docs
                analysis_data["steps"]["relevance_check"] = { "scores": [float(s) for s in scores], "relevant_count": len(final_docs) }

            # --- التعديل: إرسال المصادر إلى الواجهة الأمامية ---
            if final_docs:
                sources_data = [
                    {
                        "source": doc.metadata.get("source", "مصدر غير معروف"),
                        "content_preview": doc.page_content[:200] + "...",
                        "score": round(doc.metadata.get('relevance_score', 0), 2)
                    }
                    for doc in final_docs
                ]
                yield {"type": "sources", "content": sources_data}
            # ---------------------------------------------------

            full_answer = ""
            chain_input = {"input": question, "context": final_docs}
            chain_config = {"configurable": {"session_id": session_id}}
            
            async for chunk in self.chain_with_history.astream(chain_input, config=chain_config):
                if chunk:
                    full_answer += chunk
                    yield {"type": "chunk", "content": chunk}
            
            analysis_data["final_answer"] = full_answer.strip()

        except Exception as e:
            error_msg = f"فشل في سلسلة المعالجة: {str(e)}"
            logging.error(f"[{session_id}] {error_msg}", exc_info=True)
            analysis_data["error"] = error_msg
            try: yield {"type": "error", "content": "عذرًا، حدث خطأ فادح."}
            except Exception: pass
        finally:
            finalize_analysis(analysis_data)

agent_instance = Agent()
