# 2_backend_api/models/schemas.py
from pydantic import BaseModel
from typing import Optional

class TenantStatus(BaseModel):
    """
    يمثل حالة عميل واحد في لوحة التحكم.
    """
    tenant_id: str
    entity_name: Optional[str] = "N/A"
    status: str = "Not Processed"
    last_modified: str
    document_count: int
