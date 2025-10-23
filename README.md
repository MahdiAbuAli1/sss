# Support Service Platform

Production-ready FastAPI service that serves a multi-tenant RAG assistant backed by FAISS and reranking.

## Structure
- `2_central_api_service/agent_app/main.py`: FastAPI app and endpoints.
- `2_central_api_service/agent_app/core_logic.py`: Agent initialization and RAG pipeline.
- `1_knowledge_pipeline/`: Build vector DB artifacts.
- `3_shared_resources/vector_db/`: FAISS index files.

## Requirements
- Python 3.10+ recommended
- Ollama running locally or accessible with models configured to match `.env`

## Environment
Copy `.env.example` to `.env` and set values:
```
SUPPORT_SERVICE_API_KEY=your_strong_secret
EMBEDDING_MODEL_NAME=qwen3-embedding:0.6b
CHAT_MODEL_NAME=qwen3:4b
ALLOWED_ORIGINS=http://localhost:3000
```

## Install (production set)
```
python -m pip install --upgrade pip
# Optionally install torch for your platform first if needed
# pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.prod.txt
```

## Run
From `2_central_api_service/`:
```
uvicorn agent_app.main:app --host 0.0.0.0 --port 8000 --workers 2
```
- Health: `GET /healthz`
- Readiness: `GET /readyz`
- Tenants: `GET /tenants`
- Stream QA: `POST /ask-stream` with header `X-API-Key: <your key>`

## Notes
- `.env` is ignored by git via `.gitignore`.
- Ensure FAISS index exists at `3_shared_resources/vector_db/`.
- If deploying behind another origin, set `ALLOWED_ORIGINS`.

## Testing (optional)
- Add `pytest` smoke tests for `/`, `/tenants`, `/ask-stream`.
- Add `ruff` for linting.
