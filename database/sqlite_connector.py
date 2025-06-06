# ESTE es el que transfiere los datos desde MySQL a SQLite
import mysql.connector
import sqlite3
import json

# Configuración MySQL
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'rf_db'
}

# Conexión y lectura desde MySQL
mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
mysql_cursor = mysql_conn.cursor(dictionary=True)
mysql_cursor.execute("SELECT id_persona, nombre, apellido, correo, foto, kp FROM alumnos")
alumnos = mysql_cursor.fetchall()
mysql_cursor.close()
mysql_conn.close()

# Crear archivo SQLite y tabla (si no existe)
sqlite_conn = sqlite3.connect("personas.db")
sqlite_cursor = sqlite_conn.cursor()
sqlite_cursor.execute("""
CREATE TABLE IF NOT EXISTS alumnos (
    id_persona INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    apellido TEXT NOT NULL,
    correo TEXT,
    foto TEXT NOT NULL,
    kp TEXT
)
""")

# Insertar datos desde MySQL
for alumno in alumnos:
    sqlite_cursor.execute("""
        INSERT OR REPLACE INTO alumnos (id_persona, nombre, apellido, correo, foto, kp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        alumno['id_persona'],
        alumno['nombre'],
        alumno['apellido'],
        alumno['correo'],
        alumno['foto'],
        json.dumps(alumno['kp']) if alumno['kp'] else None
    ))

sqlite_conn.commit()
sqlite_conn.close()

print("✔ Migración completada")
