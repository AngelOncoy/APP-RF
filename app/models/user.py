class User:
    """
    Representa un usuario en la base de datos.
    """
    def __init__(self, user_id, name, last_name, email, requisitioned, image, features):
        self.user_id = user_id
        self.name = name
        self.last_name = last_name
        self.email = email
        self.requisitioned = requisitioned
        self.image = image  # Imagen en formato de bytes
        self.features = features  # Embeddings de la imagen

    def to_dict(self):
        """
        Convierte el objeto User a un diccionario para facilitar la manipulación de datos.
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "last_name": self.last_name,
            "email": self.email,
            "requisitioned": self.requisitioned,
            "image": self.image,
            "features": self.features
        }
