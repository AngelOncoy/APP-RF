import cv2
import json
import numpy as np
from skimage.feature import local_binary_pattern
from numpy.linalg import norm

# Parámetros LBP
USE_LBP_R2 = True  # Puedes desactivarlo si solo deseas usar R=1

# Detectores Haar
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ---------------------------------------
# Alínea rostro usando posición de ojos
# ---------------------------------------
def _align_face(face_gray: np.ndarray) -> np.ndarray:
    eyes = EYE_CASCADE.detectMultiScale(face_gray, 1.1, 5)
    if len(eyes) < 2:
        return cv2.resize(face_gray, (100, 100))  # Fallback sin alineamiento

    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    if cx2 < cx1:  # asegurar izquierda-derecha
        cx1, cy1, cx2, cy2 = cx2, cy2, cx1, cy1

    angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
    M = cv2.getRotationMatrix2D((face_gray.shape[1] / 2, face_gray.shape[0] / 2), angle, 1.0)
    aligned = cv2.warpAffine(face_gray, M, (face_gray.shape[1], face_gray.shape[0]))
    return cv2.resize(aligned, (100, 100))


# ----------------------------------------------------
# Preprocesamiento: detección, ecualización, alineado
# ----------------------------------------------------
def preprocess_face(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro.")

    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    # Ecualización global de histograma
    face_eq = cv2.equalizeHist(face_roi)

    # Alineamiento
    face_aligned = _align_face(face_eq)

    return face_aligned  # Devuelve rostro preprocesado, 100x100


# ----------------------------------------------------
# Extracción de características LBP
# ----------------------------------------------------
def extract_lbp_features(face: np.ndarray) -> np.ndarray:
    def lbp_hist(P, R):
        lbp = local_binary_pattern(face, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float32") / (hist.sum() + 1e-7)
        return hist

    lbp_r1 = lbp_hist(8, 1)
    lbp_r2 = lbp_hist(16, 2) if USE_LBP_R2 else np.array([], dtype="float32")

    return np.concatenate([lbp_r1, lbp_r2])


# ----------------------------------------------------
# API externa para extracción (str JSON)
# ----------------------------------------------------
def extract_face_features(image_path: str) -> str:
    face = preprocess_face(image_path)
    vec = extract_lbp_features(face)

    # Normalización L2
    vec /= (norm(vec) + 1e-7)

    return json.dumps(vec.tolist())


# ----------------------------------------------------
# Métrica: distancia Euclidiana
# ----------------------------------------------------
def euclidean_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))
