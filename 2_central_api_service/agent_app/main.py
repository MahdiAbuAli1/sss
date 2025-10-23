import os
import logging
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# استيراد المنطق الأساسي من ملفنا الآخر
# السطر الصحيح
from .core_logic import initialize_agent, get_answer_stream


# --- إعداد التسجيل (Logging) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- قراءة إعدادات الأمان من متغيرات البيئة ---
# يجب توفير SUPPORT_SERVICE_API_KEY في البيئة، وإلا سيفشل الإقلاع
EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY")
if not EXPECTED_API_KEY:
    raise RuntimeError("SUPPORT_SERVICE_API_KEY غير موجود في البيئة. قم بتعيينه قبل التشغيل.")

# --- دليل لحفظ سجلات التفاعلات ---
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
os.makedirs(LOG_DIR, exist_ok=True)

# --- حالة الجاهزية ---
IS_READY = False

# --- دورة حياة التطبيق (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 بدء تشغيل خادم الـ API...")
    # جدولة التهيئة الثقيلة لتعمل في الخلفية بدون حجب بدء التشغيل
    async def _bg_init():
        global IS_READY
        try:
            await asyncio.to_thread(initialize_agent)
            IS_READY = True
            logging.info("✅ اكتملت تهيئة الوكيل وأصبح جاهزًا للطلبات.")
        except Exception as e:
            logging.critical(f"فشل تهيئة الوكيل: {e}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_bg_init())
        logging.info("⚙️ تهيئة الوكيل ستعمل في الخلفية...")
    except Exception as e:
        logging.warning(f"تعذر جدولة تهيئة الوكيل في الخلفية: {e}")
    yield
    logging.info("🛑 إيقاف تشغيل خادم الـ API...")

# --- إنشاء تطبيق FastAPI ---
app = FastAPI(
    title="منصة الدعم الفني المركزي",
    description="واجهة برمجية للوصول إلى وكيل دعم فني متعدد العملاء.",
    version="1.0.0",
    lifespan=lifespan
)

# --- تمكين CORS اختياريًا عبر متغير البيئة ALLOWED_ORIGINS (قائمة مفصولة بفواصل) ---
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

# --- نماذج البيانات ---
class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    k_results: int = 4

# --- طبقة الأمان: التحقق من API Key ---
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != EXPECTED_API_KEY:
        logging.warning(f"محاولة وصول فاشلة باستخدام مفتاح API غير صحيح: {x_api_key}")
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")
    return x_api_key

# --- نقطة النهاية الرئيسية مع تسجيل التفاعلات ---
@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    try:
        if not IS_READY:
            raise HTTPException(status_code=503, detail="الخدمة غير جاهزة بعد. يرجى المحاولة لاحقًا.")
        # إنشاء ملف سجل لكل عميل
        log_file = os.path.join(LOG_DIR, f"{request.tenant_id}_interactions.txt")
        
        # متغير لتجميع الإجابة النهائية
        final_answer = ""

        # دالة تغليف البث لتسجيل الإجابة أثناء الإرسال
        async def generator_wrapper() -> AsyncGenerator[str, None]:
            nonlocal final_answer
            async for chunk in get_answer_stream(
                question=request.question,
                tenant_id=request.tenant_id,
                k_results=request.k_results
            ):
                final_answer += chunk
                yield chunk

            # بعد انتهاء البث، حفظ السؤال والإجابة والمصادر في الملف
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- نوع المستخدم (tenant_id) ---\n{request.tenant_id}\n")
                f.write(f"--- السؤال ---\n{request.question}\n")
                f.write(f"--- الإجابة ---\n{final_answer}\n\n")
                f.write(f"{'='*80}\n\n")

        return StreamingResponse(generator_wrapper(), media_type="text/plain")

    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع في نقطة النهاية /ask-stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="حدث خطأ داخلي في الخادم.")

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
