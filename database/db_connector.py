# db_connector.py
import mysql.connector
import json

# --- Configuración de la conexión ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'rf_db',
    'raise_on_warnings': True
}

# --- Función para conectar a la base de datos ---
def conectar_bd():
    return mysql.connector.connect(**DB_CONFIG)

# --- Función para insertar o actualizar una persona ---
def insertar_persona(nombre, apellido, correo, foto, vector_kp):
    try:
        conexion = conectar_bd()
        cursor = conexion.cursor()

        # Convertimos el vector a JSON string
        kp_json = json.dumps(vector_kp.tolist()) if vector_kp is not None else None

        insert_update_query = """
        INSERT INTO alumnos (nombre, apellido, correo, foto, kp)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            nombre = VALUES(nombre),
            apellido = VALUES(apellido),
            foto = VALUES(foto),
            kp = VALUES(kp);
        """

        cursor.execute(insert_update_query, (nombre, apellido, correo, foto, kp_json))
        conexion.commit()
        print(f"✅ Persona '{nombre} {apellido}' insertada/actualizada correctamente.")

    except mysql.connector.Error as err:
        print(f"❌ Error MySQL: {err}")

    finally:
        cursor.close()
        conexion.close()
