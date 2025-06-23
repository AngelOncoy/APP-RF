import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from app.utils.siamese_loader import get_siamese_model

# ----------------------------
# Configuración
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = get_siamese_model(device=device)

transform = transforms.Compose([
    transforms.Grayscale(),               # La red fue entrenada con imágenes en escala de grises
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ----------------------------
# Extracción de características desde imagen externa
# ----------------------------
def extract_face_features(image_path: str) -> str:
    """
    Dado un path a una imagen, detecta el rostro, lo preprocesa y extrae el embedding
    usando el modelo Siamese. Devuelve un vector JSON (str).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se puede leer la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen.")

    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face_roi, (100, 100))

    pil_image = Image.fromarray(face_resized)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.get_embedding(tensor).cpu().numpy().flatten()

    return json.dumps(embedding.tolist())


# ----------------------------
# Distancia Euclidiana
# ----------------------------
def euclidean_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))
