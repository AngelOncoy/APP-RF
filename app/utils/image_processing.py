def image_to_bytes(image_path):
    """
    Convierte una imagen a bytes para su almacenamiento en la base de datos.

    Args:
    - image_path: Ruta de la imagen a convertir.

    Returns:
    - image_bytes: La imagen convertida en bytes.
    """
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    return image_bytes
