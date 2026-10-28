# syntax=docker/dockerfile:1

# ============================================================
# Etapa 1: builder
# ============================================================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && pip install --upgrade pip==26.2.0 setuptools==84.1.0 wheel==0.46.2 \
    && rm -f /usr/local/lib/python3.12/site-packages/pip/_vendor/bom.cdx.json \
    && pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download es_core_news_md \
    && python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')" \
    && pip install --no-cache-dir waitress \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Etapa 2: final
# ============================================================
FROM python:3.12-slim-bookworm

# Actualizar pip/setuptools/wheel para eliminar CVEs de build tools
# y eliminar el SBOM interno de pip que confunde a Trivy
RUN pip install --upgrade pip==26.2.0 setuptools==84.1.0 wheel==0.46.2 \
    && rm -f /usr/local/lib/python3.12/site-packages/pip/_vendor/bom.cdx.json

# Crear usuario no privilegiado y directorios necesarios
RUN useradd -m appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    mkdir -p /usr/local/share/nltk_data && \
    mkdir -p /app && \
    chown -R appuser:appuser /home/appuser/.cache /app /usr/local/share/nltk_data

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    NLTK_DATA=/usr/local/share/nltk_data

USER appuser

RUN --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY --chown=appuser:appuser . .

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
