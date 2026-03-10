# ================================================
# Script para actualizar fichas Icecat en BD
# ================================================
import os
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ==========================================================
# Función de conexión a la base de datos
# ==========================================================
def conection_bd():
    """
    Crea y devuelve un motor de conexión a la base de datos PostgreSQL en Azure
    usando las credenciales del archivo .env.
    """
    try:
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = os.getenv("DB_NAME")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")

        engine = create_engine(
            f'postgresql://{user}:{password}@{host}:{port}/{database}'
        )
        print("✅ Conexión a BD exitosa 🚀")
        return engine

    except Exception as e:
        print("❌ Error al conectar con SQLAlchemy:", e)
        return None

# ==========================================================
# Función principal para actualizar fichas Icecat
# ==========================================================
def actualizar_fichas_icecat():
    """
    Obtiene productos sin ficha Icecat, consulta la API en paralelo,
    y guarda resultados en la base de datos omitiendo errores individuales.
    """
    
    # ==========================
    # 1. Obtener productos sin ficha
    # ==========================
    engine = conection_bd()
    if not engine:
        return

    try:
        query = "SELECT csku, cmarca FROM tbl_producto WHERE cficha_icecat IS NULL;"
        df_productos = pd.read_sql(query, engine)
        print(f"📦 Catálogo obtenido: {len(df_productos)} productos sin ficha.")
    except Exception as e:
        print("❌ Error al ejecutar la consulta inicial:", e)
        engine.dispose()
        return

    # ==========================
    # 2. Configuración API Icecat
    # ==========================
    username = os.getenv("ICECAT_USERNAME")
    app_key = os.getenv("ICECAT_APP_KEY")
    language = "es"
    base_url = "https://live.icecat.biz/api"

    resultados = []
    contador_exitos = 0
    errores_api = 0

    # ==========================
    # 3. Función de llamada API
    # ==========================
    def hacer_llamada(idx, product_code, brand):
        # Validación de marca
        if pd.isna(brand) or str(brand).strip() == "":
            print(f"⚠️ {idx} - {product_code} omitido: Marca vacía")
            return None

        params = {
            "UserName": username,
            "Language": language,
            "ProductCode": product_code,
            "Brand": brand,
            "app_key": app_key
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ {idx} - {product_code} ({brand}) OK")
                    return {"sku": product_code, "data": data}
                except ValueError:
                    print(f"❌ {idx} - {product_code} ({brand}) Error: JSON inválido")
            else:
                print(f"❌ {idx} - {product_code} ({brand}) Error HTTP {response.status_code}")
                
        except requests.RequestException as e:
            print(f"❌ {idx} - {product_code} ({brand}) Excepción de red: {e}")

        return None

    # ==========================
    # 4. Ejecutar llamadas en paralelo
    # ==========================
    print("🚀 Iniciando consultas a la API de Icecat...")
    tasks = [
        (idx, row['csku'], row['cmarca'])
        for idx, row in df_productos.iterrows()
        if pd.notna(row['csku'])
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
                contador_exitos += 1
            else:
                errores_api += 1

    # ==========================
    # 5. Actualizar base de datos iterando los resultados directamente
    # ==========================
    if not resultados:
        print("⚠️ No se obtuvieron resultados exitosos para actualizar.")
        engine.dispose()
        return

    print("💾 Iniciando guardado en base de datos...")
    conn = engine.raw_connection()
    # Habilitar autocommit para que un error en un UPDATE no aborte la transacción completa
    conn.autocommit = True 
    cursor = conn.cursor()

    query_update = "UPDATE tbl_producto SET cficha_icecat = %s WHERE csku = %s;"
    errores_sql = 0

    for r in resultados:
        sku = r["sku"]
        
        # ---------------------------------------------------------
        # Limpieza exhaustiva de caracteres nulos
        # ---------------------------------------------------------
        json_str = json.dumps(r["data"], ensure_ascii=False)
        
        # 1. Elimina el carácter nulo invisible (\x00)
        json_limpio = json_str.replace('\x00', '')
        
        # 2. Elimina el texto literal "\u0000"
        json_limpio = json_limpio.replace('\\u0000', '')
        # ---------------------------------------------------------

        try:
            cursor.execute(query_update, (json_limpio, sku))
        except Exception as e:
            print(f"❌ Error SQL al actualizar SKU {sku}: {e}")
            errores_sql += 1
            continue  # Ignora este error específico y pasa al siguiente producto

    # Cerrar conexiones
    cursor.close()
    conn.close()
    engine.dispose()

    # ==========================
    # 6. Resumen de ejecución
    # ==========================
    print("\n==================================")
    print("✅ Actualización completada")
    print(f"📊 Total procesados con éxito: {contador_exitos}")
    print(f"❌ Total errores API omitidos: {errores_api}")
    print(f"⚠️  Total errores SQL omitidos: {errores_sql}")
    print("==================================")


# ==========================================================
# Ejecución del script
# ==========================================================
if __name__ == "__main__":
    actualizar_fichas_icecat()