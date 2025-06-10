#xlxs_to_db.py
import pandas as pd
import mysql.connector
from mysql.connector import errorcode

config = {
    'user': 'root',
    'password': '123456',
    'host': 'localhost',
    'database': 'rf_db',
    'raise_on_warnings': True
}

input_excel = "../data/dataset.xlsx"

df = pd.read_excel(input_excel)

try:
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    insert_update_query = """
    INSERT INTO alumnos (nombre, apellido, correo, foto, kp)
    VALUES (%s, %s, %s, %s, %s) AS new
    ON DUPLICATE KEY UPDATE
        nombre = new.nombre,
        apellido = new.apellido,
        foto = new.foto,
        kp = new.kp;
    """

    for idx, row in df.iterrows():
        correo = row.get('Correo') if 'Correo' in df.columns else None

        if pd.isna(correo) or pd.isna(row.get('Nombre')) or pd.isna(row.get('Apellido')) or pd.isna(row.get('Foto')):
            print(f"Fila {idx} ignorada por datos incompletos o sin correo.")
            continue

        embedding = row.get('Kp') if 'Kp' in df.columns else None
        if pd.isna(embedding):
            embedding = None

        cursor.execute(insert_update_query, (
            row['Nombre'],
            row['Apellido'],
            correo,
            row['Foto'],
            embedding
        ))

    cnx.commit()
    print("Datos insertados o actualizados correctamente en la tabla alumnos.")

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Error: Usuario o contraseña incorrectos.")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Error: Base de datos no existe.")
    else:
        print(f"Error MySQL: {err}")

finally:
    cursor.close()
    cnx.close()
