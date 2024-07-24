# Imagen base oficial de Python
FROM python:3.9-slim

# Directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos y instala las dependencias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación
COPY . .

# Expone el puerto que la aplicación usará
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
