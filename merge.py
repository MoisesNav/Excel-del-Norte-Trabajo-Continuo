# DIRECTOB2B/ETL/rutina_merge.py
import os
import sys
import time
import json
import pickle
import logging
import gc
import unicodedata
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# MANIPULACIÓN DE DATOS
import pandas as pd
import numpy as np

# BASE DE DATOS
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from psycopg2.extras import execute_batch


# RED
import requests

# CONFIGURACIÓN
load_dotenv()  # Carga variables de entorno

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DB ---
def get_db_engine():
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = os.getenv("DB_NAME")
        
        if not all([user, password, host, port, database]):
            raise ValueError("Faltan variables de entorno para la BD")

        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
        return engine
    except Exception as e:
        logger.error(f"Error conectando a BD: {e}")
        return None

# --- UTILIDADES ---
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'(\d+)\s*(ml|cm|mm|kg|g)', r'\1_\2', text)
    text = re.sub(r'[^a-z0-9\s\-_\/]', ' ', text)
    return text.strip()


def normalizar_marca(marca: str) -> str:
    if pd.isna(marca):
        return None
    return re.sub(r'\s+', ' ', marca.strip()).title()

def divisora_producto_detalle(df):
    """Separa la lógica de producto maestro y sus detalles de precios."""
    # Tabla Producto
    cols_prod = ['csku', 'cnombre', 'cmarca', 'cdescripcion', 'cespecificaciones', 'cimagen', 'bestatus']
    df_prod = df[cols_prod].copy()
    
    # Timestamps
    now = datetime.now()
    df_prod['tcreate_at'] = now
    df_prod['tupdate_at'] = now
    df_prod['bestatus'] = df_prod['bestatus'].apply(lambda x: True if str(x).lower() in ['t', 'true', 'activo'] else False)

    # Tabla Precios (Pivot inverso)
    precios_list = []
    proveedores = ['ct', 'exel', 'cva', 'syscom', 'dcm']
    
    for p in proveedores:
        if f'ID_PROVEEDOR_{p}' not in df.columns: continue
        
        temp = pd.DataFrame({
            'csku': df['csku'],
            'nid_proveedor': df.get(f'ID_PROVEEDOR_{p}'),
            'ndisponibilidad': df.get(f'disponibilidad_{p}'),
            'cmoneda': df.get(f'moneda_{p}'),
            'cprecio': df.get(f'precio_{p}'),
            'clave_producto': df.get(f'clave_producto_{p}')
        })
        precios_list.append(temp)
    
    df_precios = pd.concat(precios_list, ignore_index=True).dropna(subset=['nid_proveedor'])
    
    # Normalizar monedas
    df_precios['cmoneda'] = df_precios['cmoneda'].replace({'Pesos': 'MXN', 'Dolares': 'USD'})
    
    # Filtrar consistencia
    df_precios = df_precios[df_precios['csku'].isin(df_prod['csku'])].copy()

    return df_prod, df_precios

