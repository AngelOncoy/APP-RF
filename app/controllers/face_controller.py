import json
import numpy as np
from app.services.face_recognition import extract_face_features, euclidean_distance
from app.services.db_operations import get_all_users_with_features
from app.utils.siamese_loader import get_siamese_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, threshold = get_siamese_model(device=device)


# ---------------------------
# Comparar imagen externa (Siamese Model)
# ---------------------------
def compare_external_image(image_path):
    # 1. Extraer embedding desde la imagen externa
    external_json = extract_face_features(image_path)
    external_vec = json.loads(external_json)

    # 2. Recuperar embeddings de la BD
    users = get_all_users_with_features()

    best_dist = float("inf")
    best_user = None

    for uid, name, last, email, req, vec_db in users:
        dist = euclidean_distance(external_vec, vec_db)
        if dist < best_dist:
            best_dist = dist
            best_user = (uid, name, last, email, req)

    # 3. Umbral desde modelo cargado
    from app.utils.siamese_loader import _threshold  # accede directamente al umbral cargado
    if best_dist <= _threshold and best_user:
        uid, name, last, email, req = best_user
        return {
            "match": True,
            "similarity": 1 - best_dist,  # opcional, puedes retornar solo la distancia si prefieres
            "user_data": {
                "user_id": uid,
                "name": name,
                "last_name": last,
                "email": email,
                "requisitioned": req,
            },
        }

    return {
        "match": False,
        "similarity": 1 - best_dist,
        "user_data": None
    }


# ---------------------------
# Comparación detallada (Top-k)
# ---------------------------
def compare_external_image_verbose(image_path, top_k=5):
    external_vec = json.loads(extract_face_features(image_path))
    candidates = []

    for user in get_all_users_with_features():
        uid, name, last, _, _, vec_db = user
        dist = euclidean_distance(external_vec, vec_db)
        candidates.append((dist, f"{uid}  |  {name} {last}"))

    top = sorted(candidates, key=lambda x: x[0])[:top_k]

    print("\n─ Top coincidencias ─")
    for rank, (dist, label) in enumerate(top, 1):
        print(f"{rank}.  {label:<30}  dist = {dist:.4f}")
