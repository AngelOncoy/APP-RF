#extract_features.py
import face_recognition
import pandas as pd
import json
import os

input_excel = "../data/dataset.xlsx"  # Cambia esta ruta si es necesario
RUTA_BASE = "C:/Users/Angel/OneDrive/Desktop/Percepcion/APP-RF/"  # Ajusta la ruta a tus imágenes

# Cargar el Excel
df = pd.read_excel(input_excel)

def extraer_embedding(ruta_imagen_relativa):
    ruta_imagen = os.path.join(RUTA_BASE, ruta_imagen_relativa)
    ruta_imagen = os.path.normpath(ruta_imagen)
    print(f"Procesando imagen en ruta: {ruta_imagen}")

    if not os.path.isfile(ruta_imagen):
        print(f"❌ Archivo no encontrado: {ruta_imagen}")
        return None

    try:
        imagen = face_recognition.load_image_file(ruta_imagen)
        embeddings = face_recognition.face_encodings(imagen)
        if len(embeddings) == 0:
            print(f"⚠️ No se detectó rostro en la imagen: {ruta_imagen}")
            return None
        embedding = embeddings[0].tolist()
        print(f"✅ Rostro detectado y embedding extraído para: {ruta_imagen}")
        return json.dumps(embedding)
    except Exception as e:
        print(f"❌ Error con la imagen {ruta_imagen}: {str(e)}")
        return None

# Asegurar que la columna Kp exista y acepte objetos
if 'Kp' not in df.columns:
    df['Kp'] = None
df['Kp'] = df['Kp'].astype(object)

# Aplicar función para actualizar columna Kp usando las rutas en Foto
df['Kp'] = df['Foto'].apply(extraer_embedding)

# Guardar el Excel actualizado (puedes cambiar el nombre para no sobrescribir)
df.to_excel(input_excel, index=False)

print(f"Archivo Excel actualizado correctamente: {input_excel}")
