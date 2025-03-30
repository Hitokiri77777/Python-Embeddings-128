FROM python:3.9-slim
# Estaba intentndo usar la imagen: python:3.12-alpine
# Pero tiene problemas, porque algunos paquetes se deben compilar al instalar
# Por lo que se debían incluir dependencias del sistema para compilar paquetes
# además de que es muy tardado. Y Torch, tiene problemas con los compiladores de Alpine
# Por eso la imagen es un  

# Se esta probando pytorch/pytorch:2.6.0-cpu-alpine para ver si no es necesario poner paquetes de compilación
# Por el momento se quitan, hasta probar que la imagen se pueda crear correctamente
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