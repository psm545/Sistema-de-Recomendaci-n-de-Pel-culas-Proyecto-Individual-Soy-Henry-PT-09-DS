# Imagen base oficial de Python
FROM python:3.9-slim

# Directorio de trabajo
WORKDIR /app

# Archivo de requisitos a la imagen
COPY requirements.txt requirements.txt

# Instalacion de las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia del archivo main.py y la carpeta de datos a la imagen
COPY main.py main.py
COPY data

# Puerto en el que la aplicación estará escuchando
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
