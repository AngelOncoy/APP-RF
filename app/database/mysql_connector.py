import mysql.connector
from mysql.connector import Error


def get_connection():
    """
    Obtiene la conexión a la base de datos MySQL en Railway.

    try:
        # Configuración de conexión usando el MYSQL_PUBLIC_URL de Railway
        connection = mysql.connector.connect(
            host="crossover.proxy.rlwy.net",  # Host proporcionado por Railway
            user="root",  # Usuario proporcionado por Railway
            password="XWlocoFTcvsjYhEWLePLYNCqFtjGywBt",  # Contraseña proporcionada por Railway
            database="railway",  # Nombre de la base de datos en Railway
            port=28279  # Puerto proporcionado por Railway
        )

        if connection.is_connected():
            print("Conexión exitosa a la base de datos Railway.")
            return connection
    except Error as e:
        print(f"Error de conexión: {e}")
        return None
    """

    try:
        # Configuración de conexión usando el MYSQL_local
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="123456",
            database="rf_db"
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
        print("Conexión cerrada.")
