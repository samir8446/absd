FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TWIN_CACHE_DIR=/data/cache
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY twin_engine.py app.py benchmark.py ./
RUN useradd --create-home twin && mkdir -p /data/cache && chown -R twin /data
USER twin
VOLUME ["/data"]
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
