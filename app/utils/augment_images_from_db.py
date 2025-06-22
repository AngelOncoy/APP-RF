import cv2
import numpy as np
import io
import json
from PIL import Image, ImageEnhance
from app.services.db_operations import get_all_users_with_features, get_user_image
from app.services.face_recognition import extract_face_features


# ----------------------
# Aumentar una imagen
# ----------------------
def augment_image(image: np.ndarray) -> list:
    """Recibe una imagen y retorna una lista de imágenes aumentadas"""
    aug_list = []

    # Convertir a PIL para operaciones más simples
    img_pil = Image.fromarray(image)

    # Flip horizontal
    aug_list.append(np.array(img_pil.transpose(Image.FLIP_LEFT_RIGHT)))

    # Rotaciones
    for angle in [-15, 15]:
        rotated = img_pil.rotate(angle)
        aug_list.append(np.array(rotated))

    # Brillo
    for factor in [0.7, 1.3]:
        enhancer = ImageEnhance.Brightness(img_pil)
        bright = enhancer.enhance(factor)
        aug_list.append(np.array(bright))

    # Zoom (recorte)
    h, w = image.shape
    crop = image[5:h-5, 5:w-5]
    zoomed = cv2.resize(crop, (w, h))
    aug_list.append(zoomed)

    return aug_list


# ----------------------
# Ejecutar augmentación
# ----------------------
def generate_augmented_dataset():
    users = get_all_users_with_features()
    X, y = [], []

    for user_id, _, _, _, _, _ in users:
        image_bytes = get_user_image(user_id)
        if not image_bytes:
            continue

        # Reconstruir imagen desde bytes
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        try:
            # Extraer el original
            vec = json.loads(extract_face_features_from_array(image))
            X.append(vec)
            y.append(str(user_id))

            # Generar augmentaciones
            augmented_images = augment_image(image)
            for img_aug in augmented_images:
                try:
                    vec_aug = json.loads(extract_face_features_from_array(img_aug))
                    X.append(vec_aug)
                    y.append(str(user_id))
                except Exception as e:
                    print(f"[!] Falló augmentación de {user_id}: {e}")
        except Exception as e:
            print(f"[!] Falló imagen base de {user_id}: {e}")

    return np.array(X, dtype=np.float32), np.array(y)


# ----------------------
# Adaptación: input de array directamente (sin path)
# ----------------------
def extract_face_features_from_array(image: np.ndarray) -> str:
    """Versión de extract_face_features() que acepta imagen directamente"""
    from app.services.face_recognition import FACE_CASCADE, _align_face, extract_lbp_features

    faces = FACE_CASCADE.detectMultiScale(image, 1.1, 5)
    if len(faces) == 0:
        raise ValueError("No se detectó rostro.")

    x, y, w, h = faces[0]
    face_roi = image[y:y + h, x:x + w]

    # Ecualización global
    face_eq = cv2.equalizeHist(face_roi)

    # Alineamiento
    face_aligned = _align_face(face_eq)
    vec = extract_lbp_features(face_aligned)

    vec /= (np.linalg.norm(vec) + 1e-7)
    return json.dumps(vec.tolist())


# ----------------------
# Ejecución directa
# ----------------------
if __name__ == "__main__":
    X_aug, y_aug = generate_augmented_dataset()
    print(f"✅ Dataset generado: {X_aug.shape[0]} muestras")
