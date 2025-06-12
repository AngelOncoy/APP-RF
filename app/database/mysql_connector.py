import mysql.connector
from mysql.connector import Error

def get_connection():
    """
    Obtiene la conexión a la base de datos MySQL.
    """
    try:
        connection = mysql.connector.connect(
            host="localhost",  # Cambia esto si tu base de datos está en otro host
            user="root",  # Cambia con tu usuario
            password="123456",  # Cambia con tu contraseña
            database="rf_db"  # Cambia con el nombre de tu base de datos
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error de conexión: {e}")
        return None

def close_connection(connection):
    """
    Cierra la conexión a la base de datos.
    """
    if connection and connection.is_connected():
        connection.close()
