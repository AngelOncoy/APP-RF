import numpy as np
from app.services.face_recognition import extract_face_features, cosine_similarity
from app.services.db_operations import get_all_users_with_features

def compare_external_image(image_path, similarity_threshold=0.85):
    """
    Compara una imagen externa contra todos los usuarios de la base de datos.

    Returns:
    - dict con resultado: {'match': True/False, 'user_data': {...} o None}
    """
    # 1️⃣ Extraer features de la imagen externa
    external_features = extract_face_features(image_path)

    # 2️⃣ Obtener todos los usuarios de la DB
    users = get_all_users_with_features()

    # 3️⃣ Comparar con cada uno
    best_similarity = -1
    best_user = None

    for user in users:
        user_id, name, last_name, email, requisitioned, db_features = user
        sim = cosine_similarity(external_features, db_features)

        if sim > best_similarity:
            best_similarity = sim
            best_user = user

    # 4️⃣ Decidir si es coincidencia o no
    if best_similarity >= similarity_threshold:
        user_id, name, last_name, email, requisitioned, _ = best_user
        return {
            'match': True,
            'similarity': best_similarity,
            'user_data': {
                'user_id': user_id,
                'name': name,
                'last_name': last_name,
                'email': email,
                'requisitioned': requisitioned
            }
        }
    else:
        return {
            'match': False,
            'similarity': best_similarity,
            'user_data': None
        }
