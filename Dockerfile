FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pre-rendered first screen and link-preview tags in Streamlit's index.html
RUN python prerender/patch_index.py
EXPOSE 8501
CMD ["streamlit", "run", "Multi pdf chat.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
