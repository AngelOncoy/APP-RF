#!/usr/bin/env python3
"""
Construye (o actualiza) el índice de vecinos más cercanos
usando scikit-learn y guarda:

 • app/models/nn_cosine.pkl   –  objeto NearestNeighbors(n_neighbors=1)
 • app/models/labels.pkl      –  lista de etiquetas paralela

Las imágenes deben estar en:
    data/user_photos/<identidad>/*.jpg
"""

import pickle, cv2, numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from app.services.embedder_core import align_face, embed_face

# Ruta a las fotos de usuarios (cambia si la tuya es distinta)
ROOT = Path("data/user_photos")

# Carpeta donde se guardarán indice y etiquetas
MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "models"
MODELS_DIR.mkdir(exist_ok=True)

vecs, labels = [], []

print("⏳ Recorriendo", ROOT)
for person_dir in ROOT.iterdir():
    if not person_dir.is_dir():
        continue
    for img_path in person_dir.glob("*.jpg"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print("⚠️  No se pudo leer", img_path)
            continue
        gray = align_face(bgr)
        if gray is None:
            print("⚠️  Sin rostro en", img_path)
            continue
        vec = embed_face(gray)              # ndarray (128,)
        vecs.append(vec)
        labels.append(person_dir.name)      # etiqueta = nombre de carpeta

if not vecs:
    raise RuntimeError("No se generaron embeddings: revisa la ruta ROOT")

X = np.stack(vecs).astype("float32")       # (N,128) ya L2-normalizados
nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)

pickle.dump(nn,     open(MODELS_DIR / "nn_cosine.pkl", "wb"))
pickle.dump(labels, open(MODELS_DIR / "labels.pkl",    "wb"))

print(f"✅ Índice guardado: {len(labels)} embeddings → {MODELS_DIR}")
