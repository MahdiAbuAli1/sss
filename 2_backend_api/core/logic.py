# 2_backend_api/core/logic.py
import os
import json
import datetime
from typing import List
from ..models.schemas import TenantStatus

# تحديد المسار الرئيسي لمجلدات العملاء
CLIENT_DOCS_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../4_client_docs/"))

async def get_all_tenants_status() -> List[TenantStatus]:
    """
    يمسح مجلد العملاء ويجمع معلومات الحالة لكل عميل.
    """
    tenants = []
    if not os.path.exists(CLIENT_DOCS_BASE_DIR):
        return []

    for tenant_id in os.listdir(CLIENT_DOCS_BASE_DIR):
        tenant_path = os.path.join(CLIENT_DOCS_BASE_DIR, tenant_id)
        if not os.path.isdir(tenant_path):
            continue

        # الحصول على معلومات أساسية
        last_modified_ts = os.path.getmtime(tenant_path)
        last_modified_dt = datetime.datetime.fromtimestamp(last_modified_ts).strftime('%Y-%m-%d %H:%M:%S')
        
        # حساب عدد المستندات (مع تجاهل الملفات المخفية و config)
        docs = [f for f in os.listdir(tenant_path) if os.path.isfile(os.path.join(tenant_path, f)) and not f.startswith('.') and f != 'config.json' and f != 'status.json']
        doc_count = len(docs)

        # قراءة ملف الحالة والاسم (بشكل غير متزامن إذا أمكن)
        status_file = os.path.join(tenant_path, "status.json")
        config_file = os.path.join(tenant_path, "config.json")
        
        status = "Not Processed"
        entity_name = "N/A"

        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as f:
                status_data = json.load(f)
                status = status_data.get("status", "Unknown")

        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                entity_name = config_data.get("entity_name", "N/A")

        tenants.append(TenantStatus(
            tenant_id=tenant_id,
            entity_name=entity_name,
            status=status,
            last_modified=last_modified_dt,
            document_count=doc_count
        ))
        
    return tenants
