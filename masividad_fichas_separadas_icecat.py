# ================================================
# Script para actualizar fichas Icecat en BD (Nodos Separados)
# ================================================
# Descripción:
#   - Obtiene productos sin datos de Icecat desde la BD
#   - Consume la API en paralelo usando sesiones persistentes
#   - Extrae únicamente los nodos: Image, Multimedia, GeneralInfo, CatalogObjectCloud
#   - Actualiza la tabla `tbl_producto` en bloque (batch) casteando a jsonb
#   - Ignora errores de actualización SQL y los registra
# ================================================

import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine
from psycopg2.extras import execute_batch
import logging
from dotenv import load_dotenv
import os
import sys

# CONFIGURACIÓN
load_dotenv()

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

# Utilidad para evitar errores con nodos vacíos y limpiar caracteres nulos
def safe_json_dump(val):
    if val is None:
        return None  # Se convertirá en NULL en PostgreSQL
    return json.dumps(val, ensure_ascii=False).replace('\x00', '').replace('\\u0000', '')

# ==========================================================
# Función principal para actualizar fichas Icecat
# ==========================================================
def actualizar_fichas_icecat():
    # ==========================
    # 1. Obtener productos pendientes
    # ==========================
    engine = get_db_engine()
    if engine is None:
        return

    try:
        # Optimización: Filtramos por los que no tienen Info General (puedes ajustar el WHERE)
        query = """
            SELECT csku, cmarca 
            FROM tbl_producto 
            WHERE jinfo_general_icecat IS NULL;
        """
        df_informacion = pd.read_sql(query, engine)
        print(f"📦 Catálogo obtenido: {len(df_informacion)} registros pendientes.")
    except Exception as e:
        print("❌ Error al ejecutar la consulta:", e)
        return engine.dispose()

    if df_informacion.empty:
        print("✅ No hay productos pendientes por actualizar.")
        return engine.dispose()

    # ==========================
    # 2. Configuración API Icecat
    # ==========================
    username = os.getenv("ICECAT_USERNAME")
    language = "es"
    app_key = os.getenv("ICECAT_APP_KEY")
    base_url = "https://live.icecat.biz/api"

    resultados, errores = [], []
    contador_exitos = 0

    session = requests.Session()

    # ==========================
    # 3. Función de llamada API (Optimizada para parseo selectivo)
    # ==========================
    def hacer_llamada(idx, product_code, brand):
        nonlocal contador_exitos

        if pd.isna(brand) or str(brand).strip() == "":
            errores.append({"index": idx, "sku": product_code, "brand": brand, "error": "Marca vacía"})
            return None

        params = {
            "UserName": username,
            "Language": language,
            "ProductCode": product_code,
            "Brand": brand,
            "app_key": app_key
        }

        try:
            response = session.get(base_url, params=params, timeout=10)

            if response.status_code == 200:
                try:
                    full_json = response.json()
                    # Accedemos al nodo principal "data" que contiene las etiquetas
                    data_node = full_json.get("data", {})
                    
                    # Extraemos SOLO lo que necesitamos para no saturar memoria
                    extracted_data = {
                        "Image": data_node.get("Image"),
                        "Multimedia": data_node.get("Multimedia"),
                        "GeneralInfo": data_node.get("GeneralInfo"),
                        "CatalogObjectCloud": data_node.get("CatalogObjectCloud")
                    }
                    
                    print(f"✅ {idx} - {product_code} ({brand}) OK")
                    contador_exitos += 1
                    return {"sku": product_code, "data": extracted_data}
                except ValueError:
                    error_msg = "JSON inválido"
            else:
                error_msg = f"Error HTTP {response.status_code}: {response.text}"

        except requests.RequestException as e:
            error_msg = f"Excepción de red: {e}"

        print(f"❌ {idx} - {product_code} ({brand}) - {error_msg}")
        errores.append({"index": idx, "sku": product_code, "brand": brand, "error": error_msg})
        return None

    # ==========================
    # 4. Ejecutar llamadas en paralelo
    # ==========================
    tasks = [
        (idx, row['csku'], row['cmarca'])
        for idx, row in df_informacion.iterrows()
        if not pd.isna(row['csku'])
    ]

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_data = {
            executor.submit(hacer_llamada, idx, sku, marca): idx
            for idx, sku, marca in tasks
        }

        for future in as_completed(future_to_data):
            result = future.result()
            if result:
                resultados.append(result)

    # ==========================
    # 5. Construcción de DataFrame con Columnas Separadas
    # ==========================
    df_resultados = pd.DataFrame([
        {
            "sku": r["sku"],
            "jimagen": safe_json_dump(r["data"].get("Image")),
            "jmultimedia": safe_json_dump(r["data"].get("Multimedia")),
            "jinfo": safe_json_dump(r["data"].get("GeneralInfo")),
            "jcatalog": safe_json_dump(r["data"].get("CatalogObjectCloud"))
        } for r in resultados
    ], columns=["sku", "jimagen", "jmultimedia", "jinfo", "jcatalog"])

    if df_resultados.empty:
        print("⚠️ No se obtuvieron resultados válidos de la API. Finalizando proceso.")
        engine.dispose()
        return

    # Renombramos y cruzamos para asegurar consistencia
    df_informacion = df_informacion.rename(columns={'csku': 'sku'})
    df_merged = pd.merge(df_resultados, df_informacion, how="left", on="sku")

    # ==========================
    # 6. Actualizar base de datos (Batch con jsonb)
    # ==========================
    # Preparamos las tuplas en el orden exacto del query
    updates = list(zip(
        df_merged['jimagen'],
        df_merged['jmultimedia'],
        df_merged['jinfo'],
        df_merged['jcatalog'],
        df_merged['sku']
    ))
    
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    # IMPORTANTE: Se añade ::jsonb para asegurar el casteo nativo en Postgres
    # IMPORTANTE: Se añade "jCatalogObjectCloud" entre comillas dobles
    query = """
        UPDATE tbl_producto 
        SET jimagen_icecat = %s::jsonb,
            jmultimedia_icecat = %s::jsonb,
            jinfo_general_icecat = %s::jsonb,
            "jCatalogObjectCloud" = %s::jsonb
        WHERE csku = %s;
    """
    errores_sql = []

    try:
        execute_batch(cursor, query, updates, page_size=500)
        conn.commit()
    except Exception as e_batch:
        conn.rollback()
        print("⚠️ Advertencia: Ocurrió un error en la carga masiva. Aislando errores fila por fila...")
        
        for img, multi, info, cat, sku in updates:
            try:
                cursor.execute(query, (img, multi, info, cat, sku))
                conn.commit()
            except Exception as e_row:
                conn.rollback()
                print(f"❌ Error al actualizar SKU {sku}: {e_row}")
                errores_sql.append({"sku": sku, "error": str(e_row)})

    cursor.close()
    conn.close()
    engine.dispose()

    # ==========================
    # 7. Resumen y log de errores
    # ==========================
    print("\n✅ Actualización completa")
    print(f"📊 Total respuestas exitosas procesadas: {len(resultados)}")
    print(f"❌ Total errores API: {len(errores)}")
    print(f"⚠️ Errores SQL capturados: {len(errores_sql)}")

# ==========================================================
# Ejecución del script
# ==========================================================
if __name__ == "__main__":
    actualizar_fichas_icecat()