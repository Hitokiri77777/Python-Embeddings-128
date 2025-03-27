FROM python:3.12-alpine
WORKDIR /app
ADD . /app
RUN
# Instala dependencias del sistema necesarias para compilar paquetes
RUN apk add --no-cache \
    build-base \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev

# Actualiza pip, setuptools y wheel antes de instalar los paquetes
RUN pip install --upgrade pip setuptools wheel 
# Instala las dependencias del proyecto
RUN pip install -r requirements.txt
CMD ["python","app.py"]