import cv2
import numpy as np
import os

# Cargar un clasificador Haar para la detección de rostros (libre, clásico)
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def extract_face_features(image_path):
    """
    Detecta el rostro y extrae características HOG del rostro.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen.")

    # Tomamos la primera detección
    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]

    # Redimensionamos el rostro a un tamaño fijo para HOG
    face_resized = cv2.resize(face_roi, (128, 128))

    # HOG descriptor
    hog = cv2.HOGDescriptor()
    features = hog.compute(face_resized)

    return features.flatten()

from numpy.linalg import norm

def cosine_similarity(a, b):
    """
    Calcula la similitud del coseno entre dos vectores.
    """
    return np.dot(a, b) / (norm(a) * norm(b))

