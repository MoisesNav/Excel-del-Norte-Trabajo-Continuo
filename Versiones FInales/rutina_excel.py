# DIRECTOB2B/Rutinas_por_proveedor/rutina_exel.py
import json
import os
import logging
import requests
import pandas as pd
import numpy as np
import io  # Para manejo de buffers de memoria (Carga rápida)
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# Configuración de entorno y logs
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    """Crea la conexión a la base de datos usando variables de entorno."""
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = os.getenv("DB_NAME")
        # Aseguramos que use el driver psycopg2 para el método COPY
        return create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')
    except Exception as e:
        logger.error(f"Error en la conexión a la DB: {e}")
        return None

def obtener_catalogo_exel() -> pd.DataFrame:
    """Obtiene y consolida productos, fichas técnicas e imágenes de la API de Exel."""
    api_key = os.getenv("EXEL_API_KEY")
    if not api_key:
        logger.error("API Key de Exel no encontrada en .env")
        return pd.DataFrame()
       
    base_url = "https://api01.exeldelnorte.com.mx/"
    headers = {"Authorization": api_key}
   
    def fetch_endpoint(endpoint, params=None):
        try:
            r = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            return pd.DataFrame(data.get("datos") or data.get("DATA") or [])
        except Exception as e:
            logger.warning(f"Error consultando endpoint {endpoint}: {e}")
            return pd.DataFrame()

    logger.info("Consultando API de Exel (Productos, Fichas e Imágenes)...")
    df_prod = fetch_endpoint("productos", {"sin_stock": "false"})
    if df_prod.empty: return pd.DataFrame()
   
    df_tech = fetch_endpoint("productos_fichatecnica")
    df_imgs = fetch_endpoint("imagenes")
   
    if not df_tech.empty:
        df_prod = df_prod.merge(df_tech[['sku', 'ficha_tecnica']], on='sku', how='left')
   
    if not df_imgs.empty:
        df_prod = df_prod.merge(df_imgs[['sku', 'imagenes']], on='sku', how='left')
       
    return df_prod

def procesar_catalogo_exel(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y estandariza el DataFrame."""
    if df.empty: return df

    # Columnas a eliminar según tus requerimientos
    columnas_eliminar = [
        'id', 'marca_id', 'familia_id', 'subcategoria_id',
        'categoria_id', 'codigo_sat', 'precio_oferta', 'precio_sin_oferta',
        'familia_nombre', 'subcategoria_nombre',
        'codigo_barras', 'oferta'
    ]
    df.drop(columns=columnas_eliminar, errors='ignore', inplace=True)
   
    # Mapeo y creación de columnas
    df['ID_PROVEEDOR'] = '4'
   
    rename_map = {
        'sku': 'SKU',
        'nombre': 'nombre_exel',
        'stock': 'disponibilidad_exel',
        'precio': 'precio_exel',
        'moneda': 'moneda_exel',
        'marca_nombre': 'marca_exel',
        'categoria_nombre': 'categoria_exel',
        'ficha_tecnica': 'especificaciones_exel',
        'imagenes': 'imagen_exel',
        'referencia' : 'clave_producto_exel',
        'descripcion_extendida': 'descripcion_exel'
    }
    df.rename(columns=rename_map, inplace=True)
   
    # Limpieza de SKU y serialización de JSON
    df['SKU'] = df['SKU'].astype(str).str.strip()
   
    # Opción A: Limpiar espacios en blanco y convertir vacíos a NaN
    df['descripcion_exel'] = df['descripcion_exel'].str.strip().replace('', np.nan)
   
    mask = df['descripcion_exel'].isna()
    df.loc[mask, 'descripcion_exel'] = df.loc[mask, 'nombre_exel']
   
    # Limpiar caracteres que rompen el formato (Saltos de línea y tabs)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r'[\r\n\t]+', ' ', regex=True).str.strip()
   
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    # Asegurar que los precios sean numéricos y cambiar nulos por 0
    if 'precio_exel' in df.columns:
        df['precio_exel'] = pd.to_numeric(df['precio_exel'], errors='coerce').fillna(0)

    return df

def cargar_temporal(df: pd.DataFrame):
    """Carga masiva ultra rápida usando COPY EXPERT (Modo CSV)."""
    engine = get_db_engine()
    if not engine: return
   
    try:
        # 1. Convertimos a CSV real con comillas para proteger el texto
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False, sep='|', na_rep='NULL', quoting=1) # quoting=1 es csv.QUOTE_ALL
        buffer.seek(0)
       
        # 2. Preparamos los nombres de las columnas
        columnas = ', '.join([f'"{c}"' for c in df.columns])
       
        # 3. SQL para COPY con formato CSV y delimitador personalizado
        copy_sql = f"""
            COPY temp_tbl_exel ({columnas})
            FROM STDIN
            WITH (FORMAT CSV, DELIMITER '|', NULL 'NULL')
        """
       
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE temp_tbl_exel"))
            logger.info("Tabla truncada. Iniciando flujo de datos masivo...")
           
            # Acceso al cursor raw de psycopg2
            raw_conn = conn.connection
            cursor = raw_conn.cursor()
            cursor.copy_expert(sql=copy_sql, file=buffer)
           
            logger.info(f"¡Éxito! {len(df)} registros cargados correctamente.")
           
    except Exception as e:
        logger.error(f"Fallo en carga masiva: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    logger.info("Iniciando rutina de actualización...")
    df_raw = obtener_catalogo_exel()
   
    if not df_raw.empty:
        df_final = procesar_catalogo_exel(df_raw)
        cargar_temporal(df_final)
    else:
        logger.error("No hay datos para procesar.")
