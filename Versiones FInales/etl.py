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
    return re.sub(r'\s+', ' ', str(marca).strip()).upper()

def divisora_producto_detalle(df):
    # --- 1. TABLA DE PRODUCTOS ---
    columnas_finales_productos = ['csku', 'cnombre', 'cmarca', 'cdescripcion', 'cespecificaciones', 'cimagen', 'bestatus']
    
    # Extraemos solo las columnas que sí existen en el DF original para evitar KeyError
    cols_existentes = [col for col in columnas_finales_productos if col in df.columns]
    df_tbl_productos = df[cols_existentes].copy()

    # Validar y rellenar columnas requeridas si faltan
    columnas_requeridas = ["csku", "cnombre", "cmarca", "cdescripcion", "cespecificaciones", "cimagen", "tcreate_at", "tupdate_at", "bestatus"]

    for col in columnas_requeridas:
        if col not in df_tbl_productos.columns:
            if col in ["tcreate_at", "tupdate_at"]:
                df_tbl_productos[col] = datetime.now()
            else:
                df_tbl_productos[col] = None


    # --- 2. TABLA DE PRECIOS / DETALLE (SOLO EXEL) ---
    # Creamos el DataFrame directamente con las columnas de Exel
    df_precios_filtrado = pd.DataFrame({
        'csku': df['csku'],
        # Pongo un respaldo por si la columna viene como ID_PROVEEDOR_exel o solo ID_PROVEEDOR
        'nid_proveedor': df.get('ID_PROVEEDOR_exel', df.get('ID_PROVEEDOR')), 
        'ndisponibilidad': df.get('disponibilidad_exel'),
        'cmoneda': df.get('moneda_exel'),
        'nprecio': df.get('precio_exel'),
        'cclave_producto': df.get('clave_producto_exel')
    })

    # Eliminar filas donde el ID_PROVEEDOR es nulo (reemplaza el antiguo dropna de la concatenación)
    df_precios_filtrado = df_precios_filtrado.dropna(subset=['nid_proveedor']).copy()

    # Filtrar solo los precios cuyos SKUs realmente existen en la tabla de productos
    df_precios_filtrado = df_precios_filtrado[df_precios_filtrado['csku'].isin(df_tbl_productos['csku'])]

    # Normalizar la moneda
    if 'cmoneda' in df_precios_filtrado.columns:
        df_precios_filtrado['cmoneda'] = df_precios_filtrado['cmoneda'].replace({'Pesos': 'MXN', 'Dolares': 'USD'})

    return df_tbl_productos, df_precios_filtrado

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
            SELECT tp.csku, tp.cnombre, tp.cdescripcion, tp.cmarca, tnc.nid as id_subcategoria
            FROM tbl_producto AS tp
            LEFT JOIN tbl_subcategoria AS tnc ON tnc.nid = tp.nid_subcategoria
            WHERE tp.nid_subcategoria IS NULL
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
        df_sub = pd.read_sql("SELECT nid as nid_subcategoria, cnombre_subcategoria as nombre_subcategoria FROM tbl_subcategoria", engine)
        
        df_final = df_prod.merge(df_sub, left_on="categoria_predicha", right_on="nombre_subcategoria", how="left")
        
        # Update en Batch
        updates = list(zip(df_final["id_subcategoria"], df_final["csku"]))
        updates = [(int(cat), sku) for cat, sku in updates if pd.notna(cat)]

        if updates:
            with engine.connect() as conn:
                with conn.connection.cursor() as cursor:
                    execute_batch(cursor, "UPDATE tbl_producto SET nid_subcategoria  = %s WHERE csku = %s", updates)
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
                WHERE ndisponibilidad_total = 0 OR csku NOT IN (SELECT DISTINCT csku FROM tbl_detalle_producto) or ndisponibilidad_total is NULL
            """))
            # Activar
            conn.execute(text("""
                UPDATE tbl_producto SET bestatus = 't' 
                WHERE ndisponibilidad_total > 0
            """))
            logger.info("Estatus de productos actualizado.")
    finally:
        engine.dispose()
        
def ponderacion_de_precio():
    engine = get_db_engine()
    if not engine: 
        return
    
    try:

        # tipo de cambio
        df_div = pd.read_sql(
            "SELECT divisa, precio FROM tbl_cambio_divisas", 
            engine
        )
        tc = dict(zip(df_div['divisa'], df_div['precio']))

        # precios proveedores
        query = """
            SELECT tdp.csku, tdp.cmoneda, tdp.nprecio, tdp.ndisponibilidad
            FROM tbl_detalle_producto tdp
            WHERE tdp.nprecio > 0 
            AND tdp.ndisponibilidad > 0
        """
        df = pd.read_sql(query, engine)

        # convertir a MXN
        df['precio_mxn'] = df['cmoneda'].map(tc).fillna(1) * df['nprecio']

        resultados = []

        for sku, g in df.groupby('csku'):

            precios = g['precio_mxn'].values
            disponibilidad = g['ndisponibilidad'].values

            mu = precios.mean()
            sigma = precios.std()

            # evitar división por 0
            if sigma == 0:
                sigma = mu * 0.05

            # peso gaussiano
            peso_gauss = np.exp(-((precios - mu) ** 2) / (2 * sigma ** 2))

            # combinar con disponibilidad
            peso_final = peso_gauss * disponibilidad

            costo = np.sum(precios * peso_final) / np.sum(peso_final)

            # agregar 5% extra
            costo = float(round(costo * 1.05, 2))

            disponibilidad_total = int(disponibilidad.sum())

            resultados.append((costo, disponibilidad_total, sku))

        # actualizar DB
        with engine.raw_connection() as conn:
            cursor = conn.cursor()

            execute_batch(
                cursor,
                """
                UPDATE tbl_producto 
                SET nprecio_b2b = %s,
                    ndisponibilidad_total = %s
                WHERE csku = %s
                """,
                resultados
            )

            conn.commit()

        logger.info("Ponderación gaussiana de precios finalizada.")

    except Exception as e:
        logger.error(f"Error en ponderación: {e}")

    finally:
        engine.dispose()
        
def actualizar_catalogos_db(df_old: pd.DataFrame, df_old_price: pd.DataFrame) -> None:
    """
    Actualiza las tablas tbl_producto y tbl_detalle_producto en la base de datos
    utilizando execute_batch para alto rendimiento.
    """
    engine = get_db_engine()
    if not engine: return
    try:
        # 1. Limpieza de datos: PostgreSQL no entiende np.nan, necesita None (NULL)
        # Esto es crítico porque 'cespecificaciones' y 'cimagen' tienen valores nulos en tu df_old
        df_prod_clean = df_old.replace({np.nan: None})
        df_price_clean = df_old_price.replace({np.nan: None})

        # 2. Conversión a lista de diccionarios para psycopg2
        datos_producto = df_prod_clean.to_dict('records')
        datos_precio = df_price_clean.to_dict('records')

        # 3. Query condicional para tbl_producto (Solo actualiza si hay cambios reales)
        # Utilizamos IS DISTINCT FROM para manejar comparaciones con NULLs de forma segura
        query_producto = """
            UPDATE tbl_producto
            SET 
                cnombre = %(cnombre)s,
                cmarca = %(cmarca)s,
                cdescripcion = %(cdescripcion)s,
                cespecificaciones = %(cespecificaciones)s,
                cimagen = %(cimagen)s,
                tupdate_at = CURRENT_TIMESTAMP
            WHERE csku = %(csku)s
            AND (
                cnombre IS DISTINCT FROM %(cnombre)s OR
                cmarca IS DISTINCT FROM %(cmarca)s OR
                cdescripcion IS DISTINCT FROM %(cdescripcion)s OR
                cespecificaciones IS DISTINCT FROM %(cespecificaciones)s OR
                cimagen IS DISTINCT FROM %(cimagen)s
            );
        """

        # 4. Query incondicional para tbl_detalle_producto (Actualiza sí o sí)
        query_precio = """
            UPDATE tbl_detalle_producto
            SET 
                ndisponibilidad = %(ndisponibilidad)s,
                cmoneda = %(cmoneda)s,
                nprecio = %(nprecio)s
            WHERE csku = %(csku)s;
        """

        # 5. Ejecución transaccional en lote (Batch)
        logger.info("Iniciando actualización en lote hacia la base de datos...")
        
        # Extraemos la conexión raw de psycopg2 desde el engine de SQLAlchemy
        with engine.connect() as conn:
            raw_conn = conn.connection
            try:
                with raw_conn.cursor() as cur:
                    # Actualizar tbl_producto
                    logger.info(f"Procesando {len(datos_producto)} registros para tbl_producto...")
                    execute_batch(cur, query_producto, datos_producto, page_size=1000)
                    
                    # Actualizar tbl_detalle_producto
                    logger.info(f"Procesando {len(datos_precio)} registros para tbl_detalle_producto...")
                    execute_batch(cur, query_precio, datos_precio, page_size=1000)
                
                # Si todo sale bien, hacemos commit de la transacción
                raw_conn.commit()
                logger.info("¡Actualización completada y confirmada en la base de datos!")
                
            except Exception as e:
                # Si hay un error (ej. llave foránea rota, timeout), hacemos rollback
                raw_conn.rollback()
                logger.error(f"Error durante la actualización. Se aplicó Rollback. Detalle: {e}")
                raise
    finally:
        engine.dispose()
        
def actualizar_tipo_cambio_usd():
    # Token y URL Banxico
    token = "3da738eaf30e07518304fea87b5d610f7e2e16f6fa917a3c61b7ad5a3cdcd861"
    url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43718/datos/oportuno"
    headers = {"Bmx-Token": token}

    # Consulta a la API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        try:
            dato = float(data['bmx']['series'][0]['datos'][0]['dato'])
            precio = int(dato) + (dato != int(dato))  # Ajuste del tipo de cambio
            fecha_actualizacion = datetime.now()

            engine = get_db_engine()

            update_sql = """
            UPDATE tbl_cambio_divisas
            SET 
                precio = :precio,
                fehca_actualizacion = :fecha_actualizacion
            WHERE divisa = :divisa;
            """

            insert_sql = """
            INSERT INTO tbl_cambio_divisas (divisa, precio, fehca_actualizacion)
            VALUES (:divisa, :precio, :fecha_actualizacion);
            """

            with engine.begin() as conn:
                result = conn.execute(text(update_sql), {
                    "precio": precio,
                    "fecha_actualizacion": fecha_actualizacion,
                    "divisa": "USD"
                })

                if result.rowcount == 0:
                    conn.execute(text(insert_sql), {
                        "divisa": "USD",
                        "precio": precio,
                        "fecha_actualizacion": fecha_actualizacion
                    })
                    print("✅ Registro insertado.")
                else:
                    print("✅ Registro actualizado.")

            engine.dispose()

        except Exception as e:
            print("⚠️ Error al procesar:", e)
    else:
        print("❌ Error al obtener datos:", response.status_code, response.text)

# Obtener tipos de cambio /PARA PRECIO PONDERADO
def obtener_tipos_cambio(engine):
    """
    Obtiene los tipos de cambio de monedas a MXN desde la tabla `tbl_cambio_divisas`.

    Parámetros:
    -----------
    engine : sqlalchemy.engine.base.Engine
        Conexión activa a la base de datos.

    Retorna:
    --------
    dict
        Diccionario con claves como códigos de moneda (e.g. 'USD', 'EUR') y valores de tipo de cambio (float).
    """
    query = "SELECT divisa, precio FROM tbl_cambio_divisas;"
    df_cambio = pd.read_sql(query, engine)
    tipos_cambio = dict(zip(df_cambio['divisa'], df_cambio['precio']))
    return tipos_cambio


# ==========================================================
# Ejecución del script
# ==========================================================
if __name__ == "__main__":

    logger.info("INICIANDO RUTINA MERGE")
        
    engine = get_db_engine()
    # 1. Cargar DataFrames directamente a variables
    tablas = ['temp_tbl_exel']

    for t in tablas:
        try:
            # Leemos el SQL y lo asignamos dinámicamente al nombre de la tabla
            globals()[t] = pd.read_sql(f"SELECT * FROM {t}", engine)
            logger.info(f"Cargado {t}: {len(globals()[t])} registros")
        except Exception as e:
            globals()[t] = pd.DataFrame()
            logger.error(f"Error al cargar {t}: {e}")
    
    df_final = temp_tbl_exel.copy()
    
    # Renombrar a esquema DB
    mapping = {
        'SKU': 'csku', 'nombre_exel': 'cnombre', 'categoria_exel': 'ccategoria',
        'marca_exel': 'cmarca', 'descripcion': 'cdescripcion', 
        'especificaciones_exel': 'cespecificaciones', 'imagen_exel': 'cimagen', 'descripcion_exel':'cdescripcion'
    }
    df_final.rename(columns=mapping, inplace=True)
    
     # Limpieza
    df_final.dropna(subset=['csku', 'cnombre'], inplace=True)
    df_final = df_final[df_final['cnombre'] != 'ND']
    df_final['bestatus'] = 't'
    
    df_final = df_final.replace(['NULL', 'null', 'None', 'nan'], np.nan)
    
    # Normalización final de marcas
    df_final['cmarca'] = df_final['cmarca'].apply(normalizar_marca)
    
     # Deduplicación por SKU (Optimización: Agrupar y tomar el primero válido)
    df_final = df_final.groupby('csku', as_index=False).first()
    
    # 3. Separar Master / Detalle
    df_prod, df_det = divisora_producto_detalle(df_final)
    
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
        
    actualizar_catalogos_db(df_old,df_old_price)
    
    print("Actualizando Precio del Dolar")
    actualizar_tipo_cambio_usd()
    
    # 8. Post-Procesos
    ponderacion_de_precio()

    actualizar_estatus_productos()
    
    # 9. ML
    categorizador()