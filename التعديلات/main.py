# # src/app/main.py

# import os
# import logging
# import json
# import time
# from fastapi import FastAPI, Header, HTTPException, Depends
# from fastapi.responses import StreamingResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# from typing import AsyncGenerator
# from dotenv import load_dotenv

# from .core_logic import initialize_agent, shutdown_agent, get_answer_stream

# # --- إعداد التسجيل المتقدم بصيغة JSON ---
# LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
# os.makedirs(LOG_DIR, exist_ok=True)
# log_file_path = os.path.join(LOG_DIR, "interactions.json.log")
# json_handler = logging.FileHandler(log_file_path, encoding="utf-8")
# json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
# json_handler.setFormatter(json_formatter)
# root_logger = logging.getLogger()
# if root_logger.hasHandlers():
#     root_logger.handlers.clear()
# root_logger.addHandler(json_handler)
# root_logger.setLevel(logging.INFO)

# # --- قراءة إعدادات البيئة ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"), override=True)
# EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     root_logger.info(json.dumps({"event": "startup", "detail": "بدء تشغيل خادم الـ API..."}))
#     await initialize_agent()
#     yield
#     await shutdown_agent()
#     root_logger.info(json.dumps({"event": "shutdown", "detail": "تم إيقاف تشغيل خادم الـ API."}))

# app = FastAPI(title="منصة الدعم الفني المركزي", version="2.0.0-flexible_db", lifespan=lifespan)
# app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# class QueryRequest(BaseModel):
#     question: str
#     tenant_id: str
#     k_results: int = 4

# async def verify_api_key(x_api_key: str = Header(None)):
#     if not EXPECTED_API_KEY: return
#     if not x_api_key or x_api_key.strip() != EXPECTED_API_KEY.strip():
#         raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")

# @app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
# async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
#     start_time = time.time()
    
#     async def generator_wrapper() -> AsyncGenerator[str, None]:
#         final_answer = ""
#         log_data = {"tenant_id": request.tenant_id, "question": request.question, "category": "unknown", "error": None}
        
#         try:
#             async for event in get_answer_stream(request.dict()):
#                 if event["type"] == "status":
#                     log_data["category"] = event.get("category", "unknown")
#                 elif event["type"] == "chunk":
#                     content = event["content"]
#                     final_answer += content
#                     yield content
#                 elif event["type"] == "error":
#                     content = event["content"]
#                     log_data["error"] = content
#                     yield content
#                     break
#         except Exception as e:
#             log_data["error"] = str(e)
#             root_logger.error(json.dumps({"event": "stream_error", "detail": str(e)}))
#             yield "حدث خطأ فادح أثناء معالجة طلبك."
#         finally:
#             duration = time.time() - start_time
#             log_data["duration_ms"] = int(duration * 1000)
#             log_data["answer"] = final_answer
#             root_logger.info(json.dumps(log_data, ensure_ascii=False))

#     return StreamingResponse(generator_wrapper(), media_type="text/plain; charset=utf-8")

# @app.get("/", include_in_schema=False)
# def read_root():
#     return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))

# @app.get("/tenants")
# def list_tenants():
#     clients_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../4_client_docs"))
#     if not os.path.isdir(clients_base): return {"tenants": []}
#     return {"tenants": [name for name in os.listdir(clients_base) if os.path.isdir(os.path.join(clients_base, name))]}
# ---------------------------------------------------------------------



import os
import logging
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import json # استيراد json

# --- استيراد المنطق الأساسي ---
from .core_logic import initialize_agent, get_answer_stream

# ... (باقي الإعدادات كما هي) ...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY")
if not EXPECTED_API_KEY:
    raise RuntimeError("SUPPORT_SERVICE_API_KEY غير موجود في البيئة. قم بتعيينه قبل التشغيل.")
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
os.makedirs(LOG_DIR, exist_ok=True)
IS_READY = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (دورة حياة التطبيق كما هي) ...
    logging.info(" بدء تشغيل خادم الـ API...")
    async def _bg_init():
        global IS_READY
        try:
            # تم تغييرها لتعمل بشكل غير متزامن مباشرة
            await initialize_agent()
            IS_READY = True
            logging.info(" اكتملت تهيئة الوكيل وأصبح جاهزًا للطلبات.")
        except Exception as e:
            logging.critical(f"فشل تهيئة الوكيل: {e}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_bg_init())
        logging.info(" تهيئة الوكيل ستعمل في الخلفية...")
    except Exception as e:
        logging.warning(f"تعذر جدولة تهيئة الوكيل في الخلفية: {e}")
    yield
    logging.info(" إيقاف تشغيل خادم الـ API...")

app = FastAPI(
    title="منصة الدعم الفني المركزي",
    description="واجهة برمجية للوصول إلى وكيل دعم فني متعدد العملاء.",
    version="1.0.0",
    lifespan=lifespan
)

# ... (إعدادات CORS كما هي) ...
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if allowed_origins:
    origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    k_results: int = 4

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != EXPECTED_API_KEY:
        logging.warning(f"محاولة وصول فاشلة باستخدام مفتاح API غير صحيح: {x_api_key}")
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")
    return x_api_key

# --- نقطة النهاية الرئيسية (مُعدّلة) ---
@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    if not IS_READY:
        raise HTTPException(status_code=503, detail="الخدمة غير جاهزة بعد. يرجى المحاولة لاحقًا.")

    log_file = os.path.join(LOG_DIR, f"{request.tenant_id}_interactions.log")

    async def generator_wrapper() -> AsyncGenerator[str, None]:
        final_answer = ""
        try:
            # --- تمرير المعاملات بشكل صحيح ---
            streamer = get_answer_stream(
                question=request.question,
                tenant_id=request.tenant_id,
                k_results=request.k_results
            )
            async for chunk_data in streamer:
                # التعامل مع الأخطاء التي قد تحدث أثناء البث
                if chunk_data.get("type") == "error":
                    error_content = json.dumps({"error": chunk_data["content"]}, ensure_ascii=False)
                    yield error_content
                    # يمكنك إيقاف البث هنا إذا أردت
                    return 
                
                chunk = chunk_data.get("content", "")
                final_answer += chunk
                # إرسال البيانات كـ JSON لسهولة التعامل في الواجهة الأمامية
                yield json.dumps({"chunk": chunk}, ensure_ascii=False) + "\n"

            # تسجيل التفاعل بعد اكتمال الإجابة
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- Question ---\n{request.question}\n")
                f.write(f"--- Answer ---\n{final_answer}\n")
                f.write(f"{'='*80}\n\n")

        except Exception as e:
            logging.error(f"خطأ في generator_wrapper: {e}", exc_info=True)
            # إرسال رسالة خطأ موحدة كجزء من البث
            yield json.dumps({"error": "حدث خطأ داخلي أثناء معالجة طلبك."}, ensure_ascii=False)

    return StreamingResponse(generator_wrapper(), media_type="application/x-ndjson")

# ... (باقي نقاط النهاية مثل /healthz, /readyz, /chat كما هي) ...
# ملاحظة: يجب تعديل جافاسكريبت في صفحة /chat للتعامل مع JSON بدلاً من النص العادي.


@app.get("/")
def read_root():
    return {"message": "مرحبًا بك في الواجهة البرمجية لمنصة الدعم الفني المركزي"}

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"status": "ok"}

@app.get("/readyz", response_class=JSONResponse)
def readyz():
    return {"ready": IS_READY}

@app.get("/tenants", response_class=JSONResponse)
def list_tenants():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../1_knowledge_pipeline/_processing_outputs"))
    tenants = []
    try:
        if os.path.isdir(base_path):
            for name in os.listdir(base_path):
                full = os.path.join(base_path, name)
                if os.path.isdir(full):
                    tenants.append(name)
    except Exception:
        pass
    return {"tenants": tenants}

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    html = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>واجهة المحادثة - منصة الدعم</title>
  <style>
    :root { --bg:#0f172a; --panel:#111827; --muted:#1f2937; --text:#e5e7eb; --accent:#22c55e; --accent2:#3b82f6; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:linear-gradient(120deg,#0b1020,#0a142e); color:var(--text); }
    .container { max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .card { background: rgba(17,24,39,0.75); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,0.06); border-radius:16px; overflow:hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
    header { padding:16px 20px; display:flex; gap:12px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); }
    header .title { font-weight:700; letter-spacing:.3px; }
    header .badge { margin-right:auto; font-size:12px; padding:4px 10px; border-radius:999px; background: linear-gradient(90deg,var(--accent),var(--accent2)); color:#0b1020; font-weight:700; }
    .controls { display:grid; grid-template-columns: 1.2fr 1fr 1fr; gap:10px; width:100%; }
    .controls select, .controls input { width:100%; padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,0.12); background:#0b1222; color:var(--text); outline:none; }
    .controls input::placeholder { color:#9ca3af; }
    .chat { height: 520px; overflow:auto; padding:16px; display:flex; flex-direction:column; gap:12px; background: radial-gradient(1200px 600px at 100% -10%, rgba(34,197,94,0.08), transparent 60%), radial-gradient(1200px 600px at 0% 110%, rgba(59,130,246,0.08), transparent 60%); }
    .bubble { max-width: 80%; padding:12px 14px; border-radius:12px; line-height:1.6; white-space:pre-wrap; }
    .me { align-self:flex-start; background:#0b1222; border:1px solid rgba(59,130,246,0.4); }
    .ai { align-self:flex-end; background:#0d1b2a; border:1px solid rgba(34,197,94,0.4); }
    .footer { display:flex; gap:10px; border-top:1px solid rgba(255,255,255,0.06); padding:12px; }
    .footer textarea { flex:1; resize:vertical; min-height:48px; max-height:160px; padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); background:#0b1222; color:var(--text); outline:none; }
    .footer button { background: linear-gradient(90deg,var(--accent),var(--accent2)); color:#0b1020; border:none; padding:12px 18px; border-radius:12px; font-weight:800; cursor:pointer; }
    .hint { font-size:12px; color:#9ca3af; padding: 8px 20px 16px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <header>
        <div class="title">واجهة المحادثة - منصة الدعم</div>
        <div class="badge">SAED</div>
      </header>
      <div style="padding:14px 16px;">
        <div class="controls">
          <select id="tenant"></select>
          <input id="apikey" type="password" placeholder="مفتاح API" />
          <input id="k" type="number" min="1" max="10" value="4" />
        </div>
      </div>
      <div id="chat" class="chat"></div>
      <div class="footer">
        <textarea id="msg" placeholder="اكتب سؤالك هنا..."></textarea>
        <button id="send">إرسال</button>
      </div>
      <div class="hint">تلميح: اختر النظام (المستفيد) من القائمة، ثم أدخل مفتاح API وأرسل رسالتك.</div>
    </div>
  </div>
  <script>
    const chat = document.getElementById('chat');
    const tenant = document.getElementById('tenant');
    const apikey = document.getElementById('apikey');
    const k = document.getElementById('k');
    const msg = document.getElementById('msg');
    const send = document.getElementById('send');

    async function loadTenants(){
      try{
        const res = await fetch('/tenants');
        const data = await res.json();
        tenant.innerHTML = '';
        (data.tenants || []).forEach(t => {
          const opt = document.createElement('option');
          opt.value = t; opt.textContent = t; tenant.appendChild(opt);
        });
      }catch(e){
        tenant.innerHTML = '<option value="">لا توجد أنظمة</option>';
      }
    }
    loadTenants();

    function addBubble(text, who){
      const div = document.createElement('div');
      div.className = 'bubble ' + (who==='ai'?'ai':'me');
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    async function sendMsg(){
      const question = msg.value.trim();
      if(!question) return;
      const tenantId = tenant.value;
      const key = apikey.value.trim();
      const kNum = parseInt(k.value || '4', 10);
      addBubble(question, 'me');
      msg.value = '';
      const aiDiv = document.createElement('div'); aiDiv.className = 'bubble ai'; aiDiv.textContent=''; chat.appendChild(aiDiv);
      chat.scrollTop = chat.scrollHeight;

      try{
        const res = await fetch('/ask-stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-API-Key': key },
          body: JSON.stringify({ question, tenant_id: tenantId, k_results: kNum })
        });
        if(!res.ok){
          const txt = await res.text();
          aiDiv.textContent = 'خطأ: ' + txt;
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while(true){
          const {done, value} = await reader.read();
          if(done) break;
          aiDiv.textContent += decoder.decode(value);
          chat.scrollTop = chat.scrollHeight;
        }
      }catch(e){
        aiDiv.textContent = 'تعذر الاتصال بالخادم';
      }
    }

    send.addEventListener('click', sendMsg);
    msg.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMsg(); }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
