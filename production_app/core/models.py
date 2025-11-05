# production_app/core/models.py

from pydantic import BaseModel
from typing import List, Dict

class TenantProfile(BaseModel):
    id: str
    name: str

class AnalysisLog(BaseModel):
    request_id: str
    session_id: str
    timestamp: str
    tenant_id: str
    question: str
    processing_path: str
    total_duration_ms: int
    steps: Dict
    final_answer: str
    error: str | None
