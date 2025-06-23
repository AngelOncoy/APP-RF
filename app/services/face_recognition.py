import cv2
import json
import numpy as np
from skimage.feature import local_binary_pattern, hog
from numpy.linalg import norm

# Parámetros
HOG_ORIENT = 11
HOG_CELL = (5, 5)
HOG_BLOCK = (2, 2)
USE_LBP_R2 = True

# Detectores Haar
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Alineado ocular ----------
def _align_face(face_gray: np.ndarray) -> np.ndarray:
    eyes = EYE_CASCADE.detectMultiScale(face_gray, 1.1, 5)
    if len(eyes) < 2:
        return cv2.resize(face_gray, (100, 100))

    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    if cx2 < cx1:
        cx1, cy1, cx2, cy2 = cx2, cy2, cx1, cy1

    angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
    M = cv2.getRotationMatrix2D((face_gray.shape[1] / 2, face_gray.shape[0] / 2), angle, 1.0)
    rot = cv2.warpAffine(face_gray, M, (face_gray.shape[1], face_gray.shape[0]))
    return cv2.resize(rot, (100, 100))

# ---------- Gabor features ----------
def extract_gabor_features(image: np.ndarray, ksize=21, sig=5.0, lambd=10.0, gamma=0.5, psi=0) -> np.ndarray:
    """Aplica filtros de Gabor en varias orientaciones y devuelve media + std de respuesta por filtro."""
    features = []
    for theta in np.linspace(0, np.pi, 6, endpoint=False):  # 6 orientaciones
        kernel = cv2.getGaborKernel((ksize, ksize), sig, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        mean = filtered.mean()
        std  = filtered.std()
        features.extend([mean, std])
    return np.array(features, dtype=np.float32)

# ---------- Función principal ----------
def extract_face_features(image_path: str) -> str:
    """
    Extrae vector de características: HOG + LBP + Gabor.
    Devuelve JSON serializado (str).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se puede leer la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro.")

    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]

    # Preprocesado: ecualización global + alineamiento ocular
    face_eq = cv2.equalizeHist(face_roi)
    face = _align_face(face_eq)  # 100x100

    # HOG
    hog_vec = hog(face,
                  orientations=HOG_ORIENT,
                  pixels_per_cell=HOG_CELL,
                  cells_per_block=HOG_BLOCK,
                  block_norm="L2-Hys",
                  feature_vector=True)

    # LBP
    def lbp_hist(P, R):
        lbp = local_binary_pattern(face, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float32") / (hist.sum() + 1e-7)
        return hist

    lbp_r1 = lbp_hist(8, 1)
    lbp_r2 = lbp_hist(16, 2) if USE_LBP_R2 else np.array([], dtype="float32")

    # Gabor
    gabor_vec = extract_gabor_features(face)

    # Vector final
    vec = np.concatenate([hog_vec, lbp_r1, lbp_r2, gabor_vec]).astype(np.float32)
    vec /= (norm(vec) + 1e-7)  # Normalización L2

    return json.dumps(vec.tolist())

# ---------- Métricas ----------
def cosine_similarity(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def euclidean_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))
