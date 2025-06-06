import sqlite3
import face_recognition
import numpy as np
import json

# --- Parámetro archivo SQLite ---
DB_FILE = "personas.db"

# Umbral distancia (menor = más estricto, 0.6 es un valor típico)
UMBRAL_DISTANCIA = 0.6

# --- Conexión SQLite ---
def conectar_bd():
    return sqlite3.connect(DB_FILE)

# --- Obtener vectores faciales desde SQLite ---
def obtener_personas_con_vectores():
    conexion = conectar_bd()
    conexion.row_factory = sqlite3.Row  # Permite acceder por nombre de columna
    cursor = conexion.cursor()
    cursor.execute("SELECT id_persona, nombre, apellido, correo, foto, kp FROM alumnos")
    resultados = cursor.fetchall()
    cursor.close()
    conexion.close()

    personas = []
    for row in resultados:
        try:
            vector = np.array(json.loads(row['kp']))
            personas.append({
                'id': row['id_persona'],
                'nombre': row['nombre'],
                'apellido': row['apellido'],
                'correo': row['correo'],
                'foto': row['foto'],
                'vector': vector
            })
        except Exception as e:
            print(f"⚠ Error cargando vector de persona {row['id_persona']}: {e}")
    return personas

# --- Extraer vector facial desde una imagen nueva ---
def extraer_vector_imagen(path_imagen):
    imagen = face_recognition.load_image_file(path_imagen)
    encodings = face_recognition.face_encodings(imagen)
    if len(encodings) == 0:
        raise ValueError("No se detectó rostro en la imagen.")
    return encodings[0]

# --- Buscar persona por comparación facial ---
def buscar_persona_por_imagen(ruta_imagen, umbral_distancia=UMBRAL_DISTANCIA):
    try:
        vector_entrada = extraer_vector_imagen(ruta_imagen)
    except Exception as e:
        return None, f"Error al procesar imagen de entrada: {e}"

    personas = obtener_personas_con_vectores()
    mejores_coincidencias = []

    for persona in personas:
        distancia = np.linalg.norm(vector_entrada - persona['vector'])
        if distancia < umbral_distancia:
            mejores_coincidencias.append((persona, distancia))

    if not mejores_coincidencias:
        return None, "No se encontró ninguna persona con similitud suficiente."

    # Ordenamos por distancia (menor es mejor)
    mejores_coincidencias.sort(key=lambda x: x[1])
    persona_mejor, distancia_mejor = mejores_coincidencias[0]

    # Convertimos distancia a similitud en porcentaje (opcional)
    similitud = max(0, 1 - distancia_mejor / 0.6)

    return persona_mejor, similitud

# --- Probar varios umbrales (opcional para calibración) ---
def probar_umbral(ruta_imagen_prueba, lista_umbral=[0.4, 0.5, 0.55, 0.6, 0.65, 0.7]):
    print(f"🔍 Pruebas con la imagen: {ruta_imagen_prueba}")
    for umbral in lista_umbral:
        resultado, mensaje = buscar_persona_por_imagen(ruta_imagen_prueba, umbral_distancia=umbral)
        if resultado is None:
            print(f"Umbral {umbral:.2f}: ❌ No encontrado. Mensaje: {mensaje}")
        else:
            persona, similitud = resultado, mensaje
            print(f"Umbral {umbral:.2f}: ✅ {persona['nombre']} {persona['apellido']} con similitud {similitud*100:.2f}%")

# 🔽 Ejemplo de uso:
# probar_umbral("foto_nueva.jpg")
