import json, numpy as np

# ⟡  nuevo import  ⟡
from app.services.embedder_core import align_face, embed_face

def extract_face_features(image_path: str) -> str:
    """
    Lee la imagen, alinea el rostro, obtiene el embedding 128-D y
    devuelve un JSON listo para la base de datos.
    """
    import cv2
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError("No se pudo leer la imagen")
    gray = align_face(bgr)
    if gray is None:
        raise ValueError("No se detectó rostro en la imagen")
    vec = embed_face(gray)                  # ndarray (128,)
    return json.dumps(vec.tolist())         # ➜ str JSON

# tu función original ya vale: =============================
def cosine_similarity(vec_a, vec_b):
    """
    Recibe dos listas de floats (o ndarrays) y devuelve la similitud coseno.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
