from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import pandas as pd
from typing import Optional
from unidecode import unidecode
import numpy as np
import re

app = FastAPI(docs_url=None, redoc_url=None)

# Cargar el DataFrame desde el archivo Parquet
data_path = 'data.parquet'
try:
    data = pd.read_parquet(data_path)
    data_status = {"status": "DataFrame loaded", "rows": len(data)}
    print(data_status)
except Exception as e:
    data_status = {"status": "Error loading DataFrame", "error": str(e)}
    print(data_status)

# Convertir 'release_date' a tipo datetime
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

# Extraer el día de la semana en inglés
data['day_of_week'] = data['release_date'].dt.day_name()

# Mapeo de días de la semana en inglés a español
day_translation = {
    'Monday': 'lunes',
    'Tuesday': 'martes',
    'Wednesday': 'miércoles',
    'Thursday': 'jueves',
    'Friday': 'viernes',
    'Saturday': 'sábado',
    'Sunday': 'domingo'
}

# Aplicar la traducción y normalizar (eliminar acentos y convertir a minúsculas)
data['day_of_week_es'] = data['day_of_week'].map(day_translation).apply(lambda x: unidecode(x.lower()))

# Normalizar la columna 'title'
data['title_normalized'] = data['title'].apply(lambda x: unidecode(x.lower()))

# Normalizar las columnas relevantes para facilitar la búsqueda
data['Crew_name_normalized'] = data['Crew_name'].apply(lambda x: unidecode(x.lower()) if isinstance(x, str) else '')
data['Crew_job'] = data['Crew_job'].apply(lambda x: unidecode(x.lower()) if isinstance(x, str) else '')

# Aplicar la normalización a la columna 'cast_names'
def normalize_cast_names(cast_data):
    if isinstance(cast_data, np.ndarray):
        # Si es un array de NumPy, convertirlo a una lista de strings
        names = [str(name).strip() for name in cast_data]
    elif isinstance(cast_data, str):
        # Si es una cadena, primero intentar dividir por comas
        if ',' in cast_data:
            # Eliminar corchetes y dividir por comas
            names = cast_data.strip('[]').split(',')
            # Eliminar espacios y comillas
            names = [name.strip().strip("'\"") for name in names]
        else:
            # Si no hay comas, usar el método anterior
            names = re.findall(r"'([^']*)'", cast_data)
    else:
        # Si no es ni array ni string, devolver una lista vacía
        return []
    
    return [unidecode(name.lower()) for name in names if name]

data['cast_names_normalized'] = data['cast_names'].apply(normalize_cast_names)

@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API Docs"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title="FastAPI Movies",
        version="1.0.0",
        description="API para consultas de películas",
        routes=app.routes,
    )

@app.get("/api-info")
def read_root():
    if data is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos")
    return {"message": "Bienvenido a la API de Datos de Películas", "data_status": data_status}




@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    mes_normalizado = mes.capitalize()
    peliculas_mes = data[data['month_name_es'] == mes_normalizado]
    cantidad = peliculas_mes.shape[0]
    return {"mensaje": f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes_normalizado}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dia_normalizado = unidecode(dia).lower()
    peliculas_dia = data[data['day_of_week_es'] == dia_normalizado]
    cantidad = peliculas_dia.shape[0]
    if cantidad == 0:
        return {"mensaje": f"No se encontraron películas estrenadas en los días {dia}"}
    return {"mensaje": f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}"}

@app.get("/score_titulo/{titulo_de_la_filmacion}")
def score_titulo(titulo_de_la_filmacion: str):
    titulo_normalizado = unidecode(titulo_de_la_filmacion.lower())
    pelicula = data[data['title_normalized'] == titulo_normalizado]
    if pelicula.empty:
        return {"mensaje": f"No se encontró la película con el título '{titulo_de_la_filmacion}'"}
    titulo = pelicula['title'].values[0]
    ano_estreno = int(pelicula['release_year'].values[0])
    score = pelicula['popularity'].values[0]
    return {"mensaje": f"La película '{titulo}' fue estrenada en el año {ano_estreno} con un score/popularidad de {score}"}

@app.get("/votos_titulo/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    titulo_normalizado = unidecode(titulo_de_la_filmacion.lower())
    pelicula = data[data['title_normalized'] == titulo_normalizado]
    if pelicula.empty:
        return {"mensaje": f"No se encontró la película con el título '{titulo_de_la_filmacion}'"}
    titulo = pelicula['title'].values[0]
    ano_estreno = int(pelicula['release_year'].values[0])
    total_votos = int(pelicula['vote_count'].values[0])
    promedio_votos = pelicula['vote_average'].values[0]
    if total_votos < 2000:
        return {"mensaje": f"La película '{titulo}' no cumple con el requisito de al menos 2000 valoraciones."}
    return {"mensaje": f"La película '{titulo}' fue estrenada en el año {ano_estreno}. La misma cuenta con un total de {total_votos} valoraciones, con un promedio de {promedio_votos}."}

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    nombre_actor_normalized = unidecode(nombre_actor.lower())
    peliculas_actor = data[data['cast_names_normalized'].apply(lambda actors: nombre_actor_normalized in actors)]
    if peliculas_actor.empty:
        return {"mensaje": f"No se encontró información para el actor '{nombre_actor}'"}
    cantidad_peliculas = peliculas_actor.shape[0]
    retorno_total = peliculas_actor['return'].sum()
    promedio_retorno = retorno_total / cantidad_peliculas
    return {"mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_total:.2f} con un promedio de {promedio_retorno:.2f} por filmación."}

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    nombre_director_normalized = unidecode(nombre_director.lower())
    director_records = data[(data['Crew_name_normalized'] == nombre_director_normalized) & (data['Crew_job'].str.contains('director', case=False))]
    if director_records.empty:
        not_director_records = data[(data['Crew_name_normalized'] == nombre_director_normalized)]
        if not_director_records.empty:
            crew_job = not_director_records['Crew_job'].values[0] if not_director_records['Crew_job'].notna().any() else 'desconocido'
            return {"mensaje": f"No es director, el {nombre_director} es {crew_job}."}
        else:
            return {"mensaje": f"No se encontró información para {nombre_director}"}
    peliculas_director = director_records[['title', 'release_date', 'revenue', 'budget']]
    peliculas_director['ganancia'] = peliculas_director['revenue'] - peliculas_director['budget']
    peliculas_director['release_date'] = peliculas_director['release_date'].dt.strftime('%Y-%m-%d')
    peliculas_info = []
    for _, row in peliculas_director.iterrows():
        info = (f"Película: {row['title']}, Fecha de lanzamiento: {row['release_date']}, "
                f"Retorno individual: {row['revenue']:.0f}, Costo: {row['budget']:.0f}, "
                f"Ganancia: {row['ganancia']:.0f}")
        peliculas_info.append(info)
    peliculas_list = "\n".join(peliculas_info)
    return {"mensaje": f"El Director: {nombre_director}\n\n{peliculas_list}"}
