import numpy as np

from app.database.mysql_connector import get_connection, close_connection
from app.models.user import User


def save_user_to_db(user):
    """
    Guarda los datos del usuario en la base de datos.

    Args:
    - user: Objeto User con los datos del usuario.
    """
    connection = get_connection()
    if connection:
        cursor = connection.cursor()

        query = """
        INSERT INTO users (user_id, name, last_name, email, requisitioned, image, features)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        # Convertir los datos del usuario a una tupla
        user_data = (
            user.user_id,
            user.name,
            user.last_name,
            user.email,
            user.requisitioned,
            user.image,
            user.features.tobytes()
        )

        cursor.execute(query, user_data)
        connection.commit()

        cursor.close()
        close_connection(connection)


def get_user_from_db(user_id):
    """
    Recupera los datos de un usuario de la base de datos por su user_id.

    Args:
    - user_id: El ID único del usuario.

    Returns:
    - user: Objeto User con los datos del usuario, o None si no se encuentra.
    """
    connection = get_connection()
    user = None

    if connection:
        cursor = connection.cursor()

        query = "SELECT * FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))

        result = cursor.fetchone()

        if result:
            # Crear el objeto User a partir de los datos recuperados
            user = User(result[0], result[1], result[2], result[3], result[4], result[5],
                        np.frombuffer(result[6], dtype=np.float32))

        cursor.close()
        close_connection(connection)

    return user


def get_all_users_with_features():
    """
    Recupera todos los usuarios y sus features de la base de datos.

    Returns:
    - List of (user_id, name, last_name, email, requisitioned, features)
    """
    connection = get_connection()
    users = []

    if connection:
        cursor = connection.cursor()
        query = "SELECT user_id, name, last_name, email, requisitioned, features FROM users"
        cursor.execute(query)

        results = cursor.fetchall()
        for row in results:
            user_id = row[0]
            name = row[1]
            last_name = row[2]
            email = row[3]
            requisitioned = bool(row[4])
            features = np.frombuffer(row[5], dtype=np.float32)
            users.append((user_id, name, last_name, email, requisitioned, features))

        cursor.close()
        close_connection(connection)

    return users


def get_all_users_basic():
    """
    Recupera todos los usuarios sin imagen ni features.

    Returns:
    - List of (user_id, name, last_name, email, requisitioned)
    """
    connection = get_connection()
    users = []

    if connection:
        cursor = connection.cursor()
        query = "SELECT user_id, name, last_name, email, requisitioned FROM users"
        cursor.execute(query)

        results = cursor.fetchall()
        for row in results:
            user_id = row[0]
            name = row[1]
            last_name = row[2]
            email = row[3]
            requisitioned = bool(row[4])
            users.append((user_id, name, last_name, email, requisitioned))

        cursor.close()
        close_connection(connection)

    return users


def get_user_image(user_id):
    """
    Recupera la imagen de un usuario como bytes.

    Args:
    - user_id: str

    Returns:
    - image_bytes or None
    """
    connection = get_connection()
    image_bytes = None

    if connection:
        cursor = connection.cursor()
        query = "SELECT image FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))

        result = cursor.fetchone()
        if result and result[0]:
            image_bytes = result[0]

        cursor.close()
        close_connection(connection)

    return image_bytes


def update_user(user_id, name, last_name, email, requisitioned, image_bytes=None, features=None):
    """
    Actualiza los datos de un usuario.

    Si image_bytes y features son None, no se actualizan esos campos.
    """
    connection = get_connection()

    if connection:
        cursor = connection.cursor()

        if image_bytes is not None and features is not None:
            query = """
                UPDATE users
                SET name = %s, last_name = %s, email = %s, requisitioned = %s, image = %s, features = %s
                WHERE user_id = %s
            """
            params = (name, last_name, email, requisitioned, image_bytes, features.tobytes(), user_id)

        else:
            query = """
                UPDATE users
                SET name = %s, last_name = %s, email = %s, requisitioned = %s
                WHERE user_id = %s
            """
            params = (name, last_name, email, requisitioned, user_id)

        cursor.execute(query, params)
        connection.commit()

        cursor.close()
        close_connection(connection)


def delete_user(user_id):
    """
    Elimina un usuario de la base de datos.
    """
    connection = get_connection()

    if connection:
        cursor = connection.cursor()
        query = "DELETE FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        connection.commit()

        cursor.close()
        close_connection(connection)


def get_user_profile(user_id):
    """
    Recupera los datos básicos de un usuario.

    Args:
    - user_id: str

    Returns:
    - (user_id, name, last_name, email, requisitioned) or None
    """
    connection = get_connection()
    user = None

    if connection:
        cursor = connection.cursor()
        query = "SELECT user_id, name, last_name, email, requisitioned FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))

        result = cursor.fetchone()
        if result:
            user = result

        cursor.close()
        close_connection(connection)

    return user
