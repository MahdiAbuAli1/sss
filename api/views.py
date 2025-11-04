# api/views.py

import os
import json
import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import TenantStatusSerializer

# المسار الرئيسي لمجلدات العملاء
CLIENT_DOCS_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../4_client_docs/"))

class TenantStatusList(APIView):
    """
    عرض لجلب قائمة بجميع العملاء وحالتهم.
    """
    def get(self, request, format=None):
        """
        يمسح مجلد العملاء ويجمع معلومات الحالة لكل عميل.
        """
        tenants_data = []
        if not os.path.exists(CLIENT_DOCS_BASE_DIR):
            return Response([], status=status.HTTP_200_OK)

        for tenant_id in os.listdir(CLIENT_DOCS_BASE_DIR):
            tenant_path = os.path.join(CLIENT_DOCS_BASE_DIR, tenant_id)
            if not os.path.isdir(tenant_path):
                continue

            # --- نفس المنطق الذي كتبناه سابقًا ---
            last_modified_ts = os.path.getmtime(tenant_path)
            last_modified_dt = datetime.datetime.fromtimestamp(last_modified_ts).strftime('%Y-%m-%d %H:%M:%S')
            
            docs = [f for f in os.listdir(tenant_path) if os.path.isfile(os.path.join(tenant_path, f)) and not f.startswith('.') and f not in ['config.json', 'status.json']]
            doc_count = len(docs)

            status_val = "Not Processed"
            entity_name = "N/A"

            status_file = os.path.join(tenant_path, "status.json")
            config_file = os.path.join(tenant_path, "config.json")

            if os.path.exists(status_file):
                with open(status_file, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                    status_val = status_data.get("status", "Unknown")

            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    entity_name = config_data.get("entity_name", "N/A")
            # --- نهاية المنطق المنسوخ ---

            tenants_data.append({
                'tenant_id': tenant_id,
                'entity_name': entity_name,
                'status': status_val,
                'last_modified': last_modified_dt,
                'document_count': doc_count
            })
        
        # استخدام Serializer للتحقق من صحة البيانات وتحويلها
        serializer = TenantStatusSerializer(data=tenants_data, many=True)
        serializer.is_valid(raise_exception=True) # للتحقق من أن البيانات تطابق النموذج
        return Response(serializer.data)
