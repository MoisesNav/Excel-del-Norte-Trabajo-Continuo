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

# --- FUNCIONES CORE ---

def insert_pre_ext(df_precios: pd.DataFrame):
    """Inserta precios y ajusta estructura de la tabla detalle."""
    df_precios = df_precios[df_precios['cprecio'].notna()].copy()
    if df_precios.empty:
        logger.warning("DataFrame de precios vacío. Saltando inserción.")
        return

    engine = get_db_engine()
    if not engine: return

    try:
        df_precios.to_sql('tbl_detalle_producto', engine, if_exists='replace', index=False)
        logger.info("Productos subidos con éxito.")

        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE public.tbl_detalle_producto
                ALTER COLUMN nid_proveedor TYPE int4 USING nid_proveedor::integer,
                ALTER COLUMN ndisponibilidad TYPE int8 USING ndisponibilidad::bigint,
                ADD CONSTRAINT tbl_detalle_producto_pkey PRIMARY KEY (csku, nid_proveedor),
                ADD CONSTRAINT fk_detalle_producto_proveedor FOREIGN KEY (nid_proveedor) 
                    REFERENCES public.tbl_proveedores (nid_proveedor) ON DELETE CASCADE,
                ADD CONSTRAINT fk_detalle_producto_sku FOREIGN KEY (csku) 
                    REFERENCES public.tbl_producto (csku) ON DELETE CASCADE;
                GRANT SELECT ON TABLE tbl_detalle_producto TO lector_usr;
            """))
            logger.info("Estructura de tabla ajustada y claves foráneas aplicadas.")
    except Exception as e:
        logger.error(f"Error en insert_pre_ext: {e}")
    finally:
        engine.dispose()

def unificar_productos(df_ct, df_exel, df_cva, df_syscom, df_dcm) -> pd.DataFrame:
    """
    Consolidación optimizada de catálogos. 
    Usa vectorización en lugar de iteración por filas.
    """
    logger.info("Iniciando unificación de productos...")
    
    # Renombrar IDs de proveedores
    dfs = {
        'ct': (df_ct, 'ID_Proveedor'),
        'exel': (df_exel, 'ID_PROVEEDOR'),
        'cva': (df_cva, 'ID_PROVEEDOR'),
        'syscom': (df_syscom, 'ID_PROVEEDOR'),
        'dcm': (df_dcm, 'ID_PROVEEDOR')
    }
    
    for suffix, (df, id_col) in dfs.items():
        df.rename(columns={id_col: f'ID_PROVEEDOR_{suffix}'}, inplace=True)

    # Merge masivo
    df_merged = df_ct.merge(df_exel, on="SKU", how="outer", suffixes=('_ct', '_exel')) \
        .merge(df_cva, on="SKU", how="outer", suffixes=('', '_cva')) \
        .merge(df_syscom, on="SKU", how="outer", suffixes=('', '_syscom')) \
        .merge(df_dcm, on="SKU", how="outer", suffixes=('', '_dcm'))

    # Lista de proveedores en orden de prioridad para metadatos
    provs = ['syscom', 'ct', 'exel', 'cva', 'dcm']
    
    # Coalescencia vectorizada (Mucho más rápido que bfill fila por fila)
    for col_base in ['nombre', 'categoria', 'marca', 'descripcion', 'especificaciones', 'imagen']:
        cols_to_check = []
        for p in provs:
            # Manejo de sufijos inconsistentes en el merge original
            col_name = f"{col_base}_{p}"
            # Corrección específica para errores de typo en el origen si existen (ej. descipcion_syscom)
            if col_base == 'descripcion' and p == 'syscom':
                if 'descipcion_syscom' in df_merged.columns: col_name = 'descipcion_syscom'
            
            if col_name in df_merged.columns:
                cols_to_check.append(col_name)
        
        # Crea la columna consolidada tomando el primer valor no nulo de la lista de prioridades
        df_merged[col_base] = df_merged[cols_to_check].bfill(axis=1).iloc[:, 0]

    # Selección y limpieza final
    cols_keep = ["SKU", "nombre", "categoria", "marca", "descripcion", "especificaciones", "imagen"]
    
    # Agregar columnas dinámicas de precios y disponibilidad
    for p in provs:
        cols_keep.extend([
            f'moneda_{p}', f'precio_{p}', f'clave_producto_{p}', 
            f'ID_PROVEEDOR_{p}', f'disponibilidad_{p}'
        ])
    
    # Filtrar solo columnas existentes
    cols_keep = [c for c in cols_keep if c in df_merged.columns]
    df_final = df_merged[cols_keep].copy()

    # Limpieza
    df_final.dropna(subset=['SKU', 'nombre'], inplace=True)
    df_final = df_final[df_final['nombre'] != 'ND']
    df_final['bestatus'] = 't'

    # Renombrar a esquema DB
    mapping = {
        'SKU': 'csku', 'nombre': 'cnombre', 'categoria': 'ccategoria',
        'marca': 'cmarca', 'descripcion': 'cdescripcion', 
        'especificaciones': 'cespecificaciones', 'imagen': 'cimagen'
    }
    df_final.rename(columns=mapping, inplace=True)

    # Normalización final de marcas
    df_final['cmarca'] = df_final['cmarca'].apply(normalizar_marca)

     # Deduplicación por SKU (Optimización: Agrupar y tomar el primero válido)
    df_final = df_final.groupby('csku', as_index=False).first()


    return df_final

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

def actualizar_productos_existentes(df_old: pd.DataFrame):
    """
    Actualiza productos existentes en la BD usando una tabla temporal para 
    llenar campos nulos con información nueva (COALESCE).
    """
    if df_old.empty:
        logger.info("No hay productos existentes para actualizar.")
        return

    engine = get_db_engine()
    if not engine: return

    logger.info("Iniciando actualización de productos existentes (rellenado de nulos)...")
    
    try:
        # 1. Subir a tabla temporal
        df_old.to_sql("temp_actualizacion_tbl_productos", engine, index=False, if_exists="replace")
        
        # 2. Ejecutar UPDATE masivo
        update_sql = """
        UPDATE tbl_producto AS p
        SET
            cmarca = COALESCE(p.cmarca, t.cmarca),
            cdescripcion = COALESCE(p.cdescripcion, t.cdescripcion),
            cespecificaciones = COALESCE(p.cespecificaciones, t.cespecificaciones),
            cimagen = COALESCE(p.cimagen, t.cimagen)
        FROM temp_actualizacion_tbl_productos AS t
        WHERE p.csku = t.csku
        AND (
            p.cmarca IS NULL OR
            p.cdescripcion IS NULL OR
            p.cespecificaciones IS NULL OR
            p.cimagen IS NULL
        );
        """
        
        with engine.begin() as conn:
            conn.execute(text(update_sql))
            # Opcional: Limpiar tabla temporal
            conn.execute(text("DROP TABLE IF EXISTS temp_actualizacion_tbl_productos"))
            
        logger.info("Actualización de campos nulos finalizada correctamente.")

    except Exception as e:
        logger.error(f"Error al actualizar productos existentes: {e}")
    finally:
        engine.dispose()

def categorizador():
    """Ejecuta la inferencia de la red neuronal para categorizar productos."""
    try:
        import tensorflow as tf
        from tensorflow.keras.utils import pad_sequences
    except ImportError:
        logger.error("Tensorflow no instalado.")
        return

    logger.info("Iniciando proceso de categorización neuronal...")
    engine = get_db_engine()
    if not engine: return

    try:
        # Obtener productos sin categoría
        query = """
            SELECT tp.csku, tp.cnombre, tp.cdescripcion, tp.cmarca, tnc.id_subcategoria
            FROM tbl_producto AS tp
            LEFT JOIN tbl_nueva_subcategoria AS tnc ON tnc.id_subcategoria = tp.id_nueva_subcategoria
            WHERE tp.id_nueva_subcategoria IS NULL
        """
        df_prod = pd.read_sql(query, engine)
        
        if df_prod.empty:
            logger.info("No hay productos pendientes de categorización.")
            return

        # Rutas dinámicas
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        RN_DIR = os.path.join(BASE_DIR, "Red_neuronal")
        
        path_model = os.path.join(RN_DIR, "modelo_categorias_optimizado.keras")
        path_tok = os.path.join(RN_DIR, "tokenizer.pkl")
        path_enc = os.path.join(RN_DIR, "labelencoder.pkl")

        if not os.path.exists(path_model):
            raise FileNotFoundError(f"Modelo no encontrado: {path_model}")

        # Cargar artefactos
        model = tf.keras.models.load_model(path_model)
        with open(path_tok, "rb") as f: tokenizer = pickle.load(f)
        with open(path_enc, "rb") as f: le = pickle.load(f)

        # Preprocesamiento
        df_prod["texto"] = (
            df_prod["cnombre"].fillna("") + " " + 
            df_prod["cdescripcion"].fillna("") + " " + 
            df_prod["cmarca"].fillna("")
        )
        df_prod["texto"] = df_prod["texto"].apply(normalize_text)
        
        seq = tokenizer.texts_to_sequences(df_prod["texto"])
        X = pad_sequences(seq, maxlen=200)

        # Inferencia
        preds = model.predict(X, verbose=0)
        y_classes = np.argmax(preds, axis=1)
        categorias = le.inverse_transform(y_classes)
        df_prod["categoria_predicha"] = categorias

        # Obtener IDs de subcategorías
        df_sub = pd.read_sql("SELECT id_subcategoria, nombre_subcategoria FROM tbl_nueva_subcategoria", engine)
        
        df_final = df_prod.merge(df_sub, left_on="categoria_predicha", right_on="nombre_subcategoria", how="left")
        
        # Update en Batch
        updates = list(zip(df_final["id_subcategoria_y"], df_final["csku"]))
        updates = [(int(cat), sku) for cat, sku in updates if pd.notna(cat)]

        if updates:
            with engine.connect() as conn:
                with conn.connection.cursor() as cursor:
                    execute_batch(cursor, "UPDATE tbl_producto SET id_nueva_subcategoria = %s WHERE csku = %s", updates)
                    conn.connection.commit()
            logger.info(f"Categorizados {len(updates)} productos.")

    except Exception as e:
        logger.error(f"Error en categorizador: {e}")
    finally:
        engine.dispose()
        tf.keras.backend.clear_session()
        gc.collect()

def actualizar_estatus_productos():
    engine = get_db_engine()
    if not engine: return
    try:
        with engine.begin() as conn:
            # Desactivar
            conn.execute(text("""
                UPDATE tbl_producto SET bestatus = 'f' 
                WHERE disponibilidad_total = 0 OR csku NOT IN (SELECT DISTINCT csku FROM tbl_detalle_producto)
            """))
            # Activar
            conn.execute(text("""
                UPDATE tbl_producto SET bestatus = 't' 
                WHERE disponibilidad_total > 0 OR csku IN (SELECT DISTINCT csku FROM tbl_detalle_producto)
            """))
            logger.info("Estatus de productos actualizado.")
    finally:
        engine.dispose()

def ponderacion_de_precio():
    engine = get_db_engine()
    if not engine: return
    try:
        df_div = pd.read_sql("SELECT divisa, precio FROM tbl_cambio_divisas", engine)
        tc = dict(zip(df_div['divisa'], df_div['precio']))
        
        query = """
            SELECT tdp.csku, tdp.cmoneda, tdp.cprecio, tdp.ndisponibilidad
            FROM tbl_detalle_producto tdp
            WHERE tdp.cprecio > 0 AND tdp.ndisponibilidad > 0
        """
        df = pd.read_sql(query, engine)
        
        df['precio_mxn'] = df['cmoneda'].map(tc).fillna(1) * df['cprecio']
        df['ponderado'] = df['precio_mxn'] * df['ndisponibilidad']
        
        grouped = df.groupby('csku').agg({
            'ponderado': 'sum',
            'ndisponibilidad': 'sum'
        }).reset_index()
        
        grouped['costo'] = (grouped['ponderado'] / grouped['ndisponibilidad']).round(2)
        
        updates = list(zip(grouped['costo'], grouped['ndisponibilidad'], grouped['csku']))
        
        with engine.raw_connection() as conn:
            cursor = conn.cursor()
            execute_batch(cursor, 
                "UPDATE tbl_producto SET precio_b2b = %s, disponibilidad_total = %s WHERE csku = %s", 
                updates)
            conn.commit()
        logger.info("Ponderación de precios finalizada.")

    except Exception as e:
        logger.error(f"Error en ponderación: {e}")
    finally:
        engine.dispose()

# --- MAIN FLOW ---
if __name__ == "__main__":
    logger.info("INICIANDO RUTINA MERGE")
    
    engine = get_db_engine()
    if engine:
        try:
            # 1. Cargar DataFrames Temporales
            tablas = ['temp_tbl_ct', 'temp_tbl_exel', 'temp_tbl_cva', 'temp_tbl_syscom', 'temp_tbl_dcm']
            data = {}
            for t in tablas:
                try:
                    data[t] = pd.read_sql(f"SELECT * FROM {t}", engine)
                    logger.info(f"Cargado {t}: {len(data[t])} registros")
                except Exception:
                    data[t] = pd.DataFrame()
            
            # 2. Unificar
            df_uni = unificar_productos(
                data['temp_tbl_ct'], data['temp_tbl_exel'], 
                data['temp_tbl_cva'], data['temp_tbl_syscom'], data['temp_tbl_dcm']
            )
            
            # 3. Separar Master / Detalle
            df_prod, df_det = divisora_producto_detalle(df_uni)
            
            # 4. Obtener Productos Existentes en BD
            existing_skus = pd.read_sql("SELECT csku FROM tbl_producto", engine)['csku'].tolist()
            existing_skus_set = set(existing_skus)
            
            # Crear máscara una sola vez (más eficiente)
            mask_new = ~df_prod['csku'].isin(existing_skus_set)

            # Separar productos
            df_new = df_prod[mask_new]
            df_old = df_prod[~mask_new]

            # Separar detalle precio usando la misma lógica
            mask_new_price = df_det['csku'].isin(df_new['csku'])

            df_new_price = df_det[mask_new_price]
            df_old_price = df_det[~mask_new_price]

            # Insertar nuevos productos
            if not df_new.empty:
                df_new.to_sql('tbl_producto', engine, if_exists='append', index=False)
                logger.info(f"Insertados {len(df_new)} nuevos productos.")

            # Insertar precios de productos nuevos
            if not df_new_price.empty:
                df_new_price.to_sql('tbl_detalle_producto', engine, if_exists='append', index=False)
                logger.info(f"Insertados {len(df_new_price)} nuevos detalles de precio.")
                
            
            
            
            # 6. Actualizar Existentes (Llenado de campos nulos)
            # Esta función recupera la lógica de 'procesar_en_db' que solicitaste
            if not df_old.empty:
                actualizar_productos_existentes(df_old)
            
            # 7. Actualizar Detalle
            insert_pre_ext(df_det)
            
            # 8. Post-Procesos
            ponderacion_de_precio()
            actualizar_estatus_productos()
            
            # 9. ML
            categorizador()
            
        except Exception as e:
            logger.critical(f"Fallo crítico en rutina merge: {e}")
        finally:
            engine.dispose()
            gc.collect()
            logger.info("RUTINA MERGE FINALIZADA")