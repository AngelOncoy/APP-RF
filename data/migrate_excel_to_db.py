import pandas as pd
import cv2
import numpy as np
import os
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
        user_id = str(row["ID"])
        name = row["Nombre"]
        last_name = row["Apellido"]
        email = row["Correo"]

        # La ruta es relativa, construimos path absoluto si hace falta
        image_path = os.path.join( row["Foto"].replace("\\", os.sep))

        # Leer imagen como bytes
        image_bytes = image_to_bytes(image_path)

        # Extraer características del rostro
        features = extract_face_features(image_path)

        # Crear objeto User
        user = User(
            user_id=user_id,
            name=name,
            last_name=last_name,
            email=email,
            requisitioned=False,  # Por defecto False
            image=image_bytes,
            features=features
        )

        # Guardar en la base de datos
        save_user_to_db(user)

        print(f"[OK] Usuario {user_id} migrado correctamente.")

    except Exception as e:
        print(f"[ERROR] Usuario {row['ID']} no pudo ser migrado: {e}")
