import pandas as pd
import cv2
import numpy as np
import os
import json
from app.models.user import User
from app.services.db_operations import save_user_to_db
from app.services.face_recognition import extract_face_features
from app.utils.image_processing import image_to_bytes

# Ruta del Excel
excel_path = "dataset.xlsx"

# Leer el Excel
df = pd.read_excel(excel_path)

# Procesar cada fila
for index, row in df.iterrows():
    try:
        user_id = str(row["ID"]).strip()
        name = str(row["Nombre"]).strip()
        last_name = str(row["Apellido"]).strip()
        email = str(row["Correo"]).strip()

        # Construir ruta absoluta desde relativa
        image_path = os.path.join(row["Foto"].replace("\\", os.sep))

        # Leer imagen como bytes
        image_bytes = image_to_bytes(image_path)

        # Extraer características
        features_vector = extract_face_features(image_path)

        # Convertir a JSON
        features_json = json.dumps(features_vector)

        # Crear objeto User con features como string JSON
        user = User(
            user_id=user_id,
            name=name,
            last_name=last_name,
            email=email,
            requisitioned=False,  # Por defecto
            image=image_bytes,
            features=features_json  # 🔁 ahora string tipo JSON
        )

        # Guardar en la base de datos
        save_user_to_db(user)

        print(f"[OK] Usuario {user_id} migrado correctamente.")

    except Exception as e:
        print(f"[ERROR] Usuario {row['ID']} no pudo ser migrado: {e}")
