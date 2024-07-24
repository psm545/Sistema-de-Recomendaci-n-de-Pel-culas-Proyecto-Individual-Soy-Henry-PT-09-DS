# Movie Data API

## Introducción

Movie Data API es un proyecto desarrollado con FastAPI y Docker para proporcionar un servicio web que consulta información sobre películas a partir de un dataset. Utilizando FastAPI, se crea una API RESTful que permite realizar varias consultas sobre el dataset de películas. Este proyecto también incluye la implementación de la API en Render, una plataforma de despliegue en la nube.

## Objetivos

1. **Desarrollar una API RESTful**: Crear una API utilizando FastAPI que permita realizar consultas sobre un dataset de películas.
2. **Consultar datos**: Implementar endpoints que permitan obtener información específica del dataset, como la cantidad de filmaciones por mes, la cantidad de filmaciones por día, el score de una película por título, y más.
3. **Desplegar la API**: Implementar el proyecto en Render para que sea accesible públicamente y pueda manejar peticiones en un entorno de producción.

## Estructura del Proyecto

El proyecto está estructurado de la siguiente manera:

- `main.py`: Archivo principal que contiene la implementación de la API en FastAPI. Define los endpoints y las funciones que procesan las solicitudes.
- `data/`: Carpeta que contiene el archivo `data.parquet`, el cual es el dataset utilizado por la API.
- `Dockerfile`: Archivo de configuración para construir la imagen Docker del proyecto.
- `requirements.txt`: Archivo que lista las dependencias necesarias para el proyecto.
- `README.md`: Este archivo de documentación.

## Procesos

### 1. Preparación del Entorno

1. **Instalación de Dependencias**: Asegúrate de tener instaladas las dependencias necesarias. Puedes instalar las dependencias utilizando el archivo `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

2. **Construcción de la Imagen Docker**: Utiliza el archivo `Dockerfile` para construir una imagen Docker del proyecto.

    ```bash
    docker build -t movie-data-api .
    ```

3. **Ejecución del Contenedor Docker**: Ejecuta un contenedor a partir de la imagen construida.

    ```bash
    docker run -p 8000:8000 movie-data-api
    ```

### 2. Implementación en Render

1. **Crear una Cuenta en Render**: Regístrate o inicia sesión en [Render](https://render.com/).

2. **Crear un Nuevo Servicio**: En el panel de control de Render, crea un nuevo servicio web. Selecciona "Web Service" y sigue las instrucciones para conectar tu repositorio de Git.

3. **Configurar el Despliegue**:
   - **Docker**: En la configuración del servicio, elige "Docker" como método de despliegue.
   - **Ruta de la Imagen Docker**: Proporciona el Dockerfile del proyecto y cualquier configuración adicional que requiera Render.

4. **Sube el Repositorio**: Asegúrate de que el repositorio de Git incluya el archivo `data.parquet` en la carpeta adecuada y los archivos de configuración necesarios para que Render pueda construir y desplegar el contenedor correctamente.

5. **Desplegar el Servicio**: Render construirá la imagen Docker y desplegará el servicio. Una vez completado, obtendrás una URL pública para tu API.

## Conclusiones

El proyecto Movie Data API proporciona una solución robusta para consultar datos de películas a través de una API RESTful. Utilizando FastAPI y Docker, el proyecto es escalable y fácil de mantener. La implementación en Render asegura que el servicio esté disponible de manera confiable en un entorno de producción, permitiendo a los usuarios realizar consultas en tiempo real sobre el dataset de películas.

## Contacto

Para más información o preguntas, puedes contactar al desarrollador del proyecto:

- **Nombre**: Felipe Orlando Amezquita Samudio
- **Email**: psm545@aol.com
