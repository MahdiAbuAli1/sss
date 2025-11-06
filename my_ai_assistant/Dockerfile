# Dockerfile لمشروعك
FROM ollama/ollama:latest

# نسخ ملفات المشروع إلى الحاوية
COPY . /app
WORKDIR /app

# تثبيت مكتبات Python المطلوبة
RUN pip install --no-cache-dir -r requirements.txt

# نقطة البداية لتشغيل مشروعك
CMD ["python3", "project_core/api/main.py"]
