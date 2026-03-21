FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV MEMORY_DB_PATH=/data/memory.db
ENV MEMORY_EMBEDDING_MODE=local

VOLUME /data

EXPOSE 8484

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8484/health')" || exit 1

CMD ["memory-layer", "serve", "--host", "0.0.0.0", "--port", "8484"]
