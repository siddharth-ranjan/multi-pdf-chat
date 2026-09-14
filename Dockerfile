FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag.py server.py ./
COPY web ./web
EXPOSE 8501
# A single worker: chats and their indexes live in this process's memory
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8501", "--proxy-headers", "--forwarded-allow-ips", "*"]
