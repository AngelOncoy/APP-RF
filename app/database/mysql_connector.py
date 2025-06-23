import mysql.connector
from mysql.connector import Error


def get_connection():
    """
    Obtiene la conexión a la base de datos MySQL en Railway.
"""
    try:
        # Configuración de conexión usando el MYSQL_PUBLIC_URL de Railway
        connection = mysql.connector.connect(
            host="gondola.proxy.rlwy.net",  # Host externo (Railway)
            user="root",  # Usuario
            password="EneGckizvvanyMXiZtHvrsfSDqvtmfRI",  # Contraseña
            database="railway",  # Base de datos
            port=42075
        )

        if connection.is_connected():
            print("Conexión exitosa a la base de datos Railway.")
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
        print("Conexión cerrada.")
