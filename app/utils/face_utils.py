import cv2
import numpy as np
import torch
from torchvision import transforms


# --- Preprocesamiento: leer imagen, escalar, convertir a tensor y normalizar ---
def preprocess_image(img_path: str, model_backbone, device='cpu'):
    """
    Devuelve el embedding (vector de características) para una imagen dada usando un modelo siamese.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer imagen: {img_path}")

    img = cv2.resize(img, (100, 100))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 1, 100, 100)

    model_backbone.eval()
    with torch.no_grad():
        embedding = model_backbone(img_tensor).cpu().numpy().flatten()
    return embedding


# --- Distancia euclidiana ---
def euclidean_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))
