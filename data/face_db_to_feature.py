# rellenar_features_from_db_v2.py
"""
Versión para modelo Siamese V2
---------------------------------
• Carga el modelo `siamese_face_modelV2.pth`
• Para cada usuario en la BD:
   - reconstruye imagen desde bytes
   - extrae vector con modelo V2 (embedding)
   - lo guarda como JSON en la BD
"""

import json
from PIL import Image
import cv2
import torch
import numpy as np

from torchvision import transforms
from app.database.mysql_connector import get_connection, close_connection
from app.utils.siamese_loader import load_siamese_model

# --- Configuración
DEVICE = "cpu"  # o "cuda" si estás en GPU
MODEL_PATH = "../app/utils/siamese_face_modelV2.pth"
THRESHOLD_PATH = "../app/utils/optimal_thresholdV2.json"

# --- Transformaciones para el modelo
transform = transforms.Compose([
    transforms.Grayscale(),               # Convertir a 1 canal
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Cargar modelo y umbral
model, _ = load_siamese_model(MODEL_PATH, THRESHOLD_PATH, device=DEVICE)

def preprocess_and_embed(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar imagen desde bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro")

    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]

    # Convertir a RGB primero (por seguridad) y luego a PIL
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    face_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.get_embedding(face_tensor).cpu().numpy().flatten()
        return json.dumps(embedding.tolist())

def regenerate_features_with_model():
    conn = get_connection()
    if not conn:
        print("❌ Error al conectar a la BD.")
        return

    cur = conn.cursor()
    cur.execute("SELECT user_id, image FROM users")
    rows = cur.fetchall()

    total, ok, fail = len(rows), 0, 0

    for user_id, img_bytes in rows:
        try:
            features_json = preprocess_and_embed(img_bytes)
            cur.execute("UPDATE users SET features = %s WHERE user_id = %s", (features_json, user_id))
            conn.commit()
            ok += 1
        except Exception as e:
            print(f"[❌ ERROR] ID {user_id}: {e}")
            fail += 1

    cur.close()
    close_connection(conn)
    print(f"✅ Embeddings actualizados: {ok} ✔ · Errores: {fail} ✘ · Total: {total}")

if __name__ == "__main__":
    regenerate_features_with_model()
