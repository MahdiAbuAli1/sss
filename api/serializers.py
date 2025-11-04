# api/serializers.py  (ملف جديد)

from rest_framework import serializers

class TenantStatusSerializer(serializers.Serializer):
    """
    يقوم بتحويل بيانات حالة العميل إلى JSON.
    """
    tenant_id = serializers.CharField(max_length=100)
    entity_name = serializers.CharField(max_length=100, default="N/A")
    status = serializers.CharField(max_length=50, default="Not Processed")
    last_modified = serializers.CharField(max_length=100)
    document_count = serializers.IntegerField()


