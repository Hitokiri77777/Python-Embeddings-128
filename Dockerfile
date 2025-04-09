#Etapa 1. Sirve para copilar, descargar e instalar lo necesario
FROM python:3.9-slim-buster AS builder

WORKDIR /app
COPY requirements.txt .

# Instala dependencias del sistema necesarias para compilar paquetes: 1,2,3
# Actualiza pip, setuptools y wheel antes de instalar los paquetes: 4
# Instalar dependencias necesarias (sin soporte CUDA): 5
# Instala las dependencias del proyecto: 6
# Descargar el modelo de spaCy: 7
# Instala el paquete Waitress: 8
# Se descarga el modelo 'paraphrase-multilingual-MiniLM-L12-v2' desde Hugging Face y se guarda en caché: 9
# Se desinstalan compiladores usados y sus dependencias. Ya no hacen falta: 10, 11, 12
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && pip install --upgrade pip setuptools \
    && pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download es_core_news_md \
    && pip install --no-cache-dir waitress \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')" \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && apt-get clean 

# Etapa 2. La que realmente se usará
FROM python:3.9-slim-buster

WORKDIR /app

# Copiar solo lo necesario desde la etapa de construcción, la 1
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/torch/sentence_transformers /root/.cache/torch/sentence_transformers

# Copiar código de la aplicación
COPY . .

# Crear y cambiar a un usuario no privilegiado
RUN useradd -m appuser && chown -R appuser:appuser /app /root/.cache
USER appuser

# Configurar variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Exponer puerto explícitamente
EXPOSE 5000

# Ejecutar la aplicación Flask con  "Waitress" que es más robusto:
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]