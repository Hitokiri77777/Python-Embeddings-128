#Verifica la arquitectura de tus servidores/contenedores  (x86 o ARM)
#Si cuentas con GPU, siempre será mejor usarla. Este proyecto no intenta usarla
FROM pytorch/pytorch:2.6.0-cpu-alpine
# Estaba intentndo usar la imagen: python:3.12-alpine
# Pero tiene problemas, porque algunos paquetes se deben compilar al instalar
# Por lo que se debían incluir dependencias del sistema para compilar paquetes
# además de que es muy tardado.

# Se esta probando pytorch/pytorch:2.6.0-cpu-alpine para ver si no es necesario poner paquetes de compilación
# Por el momento se quitan, hasta probar que la imagen se pueda crear correctamente
WORKDIR /app
ADD . /app
RUN
# # Instala dependencias del sistema necesarias para compilar paquetes
# RUN apk add --no-cache \
#     build-base \
#     cmake \
#     openblas-dev \
#     linux-headers \

# Actualiza pip, setuptools y wheel antes de instalar los paquetes
RUN pip install --upgrade pip setuptools wheel 
# Instala las dependencias del proyecto
RUN pip install -r requirements.txt
# Ejecuta nuestra aplicación
CMD ["python","app.py"]