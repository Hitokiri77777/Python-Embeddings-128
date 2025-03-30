FROM python:3.9-slim
WORKDIR /app
ADD . /app
RUN
# # Instala dependencias del sistema necesarias para compilar paquetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Actualiza pip, setuptools y wheel antes de instalar los paquetes
RUN pip install --upgrade pip setuptools
# Instalar dependencias necesarias (sin soporte CUDA)
RUN pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers
# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt
# Descargar el modelo de spaCy
RUN python -m spacy download es_core_news_md
# Ejecuta nuestra aplicación
CMD ["python","app.py"]