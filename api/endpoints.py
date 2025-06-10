import json
import uuid

import mysql
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os

# Importamos los modelos y funciones que ya tienes
from schemas.schema import PersonaRespuesta
from scripts.Reconocimiento_Facial import buscar_persona_por_imagen, extraer_vector_imagen, \
    obtener_personas_con_vectores
from database.db_connector import insertar_persona, conectar_bd

# Declaramos el router
router = APIRouter()

# --- Endpoint /comparar ---
@router.post("/comparar")
async def comparar_imagen(file: UploadFile = File(...)):
    try:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            contenido = await file.read()
            f.write(contenido)

        # --- Paso 1: verificar si hay rostro ---
        try:
            _ = extraer_vector_imagen(temp_image_path)  # Solo validar que hay rostro
        except ValueError as ve:
            print(f"❌ [COMPARAR] No se detectó rostro: {ve}")
            return JSONResponse(
                status_code=422,
                content={
                    "mensaje": str(ve),
                    "agregar": True
                },
                headers={"Content-Type": "application/json"}  # 👈 aseguramos que sea application/json
            )

        # --- Paso 2: buscar persona ---
        resultado = buscar_persona_por_imagen(temp_image_path)

        if resultado is None or resultado[0] is None:
            print(f"⚠️ [COMPARAR] Resultado: SIN COINCIDENCIA - agregar nueva persona")
            return JSONResponse(
                status_code=200,  # 👈 importante: si hay rostro pero no coincidencia, mandamos 200
                content={
                    "mensaje": "No se encontró ninguna persona con similitud suficiente.",
                    "agregar": True
                },
                headers={"Content-Type": "application/json"}
            )

        persona_mejor, similitud = resultado[0], resultado[1]

        print(f"✅ [COMPARAR] Resultado: PERSONA ENCONTRADA - {persona_mejor['nombre']} {persona_mejor['apellido']} (Similitud {similitud*100:.2f}%)")

        return JSONResponse(
            status_code=200,
            content={
                "mensaje": f"Persona reconocida con {similitud*100:.2f}%",
                "agregar": False,
                "datos": {
                    "id": persona_mejor["id"],
                    "nombre": persona_mejor["nombre"],
                    "apellido": persona_mejor["apellido"],
                    "correo": persona_mejor["correo"],
                    "foto": persona_mejor["foto"],
                    "similitud": similitud * 100
                }
            },
            headers={"Content-Type": "application/json"}
        )

    except Exception as e:
        print(f"❌ [COMPARAR] Error inesperado: {e}")
        return JSONResponse(
            status_code=500,
            content={"mensaje": f"Error inesperado al procesar la imagen: {e}"},
            headers={"Content-Type": "application/json"}
        )

    finally:
        # Limpiar imagen temporal
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")

# --- Endpoint /agregar_persona ---
@router.post("/agregar_persona")
async def agregar_persona(
    nombre: str = Form(...),
    apellido: str = Form(...),
    correo: str = Form(...),
    imagen: UploadFile = File(...)
):
    try:
        # --- 1. Guardar imagen en carpeta ---
        os.makedirs("fotos", exist_ok=True)
        extension = imagen.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{extension}"
        path_guardado = os.path.join("fotos", filename)

        with open(path_guardado, "wb") as f:
            contenido = await imagen.read()
            f.write(contenido)

        # --- 2. Extraer vector facial ---
        try:
            vector = extraer_vector_imagen(path_guardado)
        except ValueError as ve:
            print(f"❌ [AGREGAR_PERSONA] No se detectó rostro: {ve}")
            return JSONResponse(status_code=422, content={"mensaje": str(ve)})

        vector_json = json.dumps(vector.tolist())

        # --- 3. Insertar en DB ---
        conexion = conectar_bd()
        cursor = conexion.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO alumnos (nombre, apellido, correo, foto, kp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (nombre, apellido, correo, path_guardado, vector_json)
            )
            conexion.commit()
            print(f"✅ [AGREGAR_PERSONA] Persona agregada: {nombre} {apellido} ({correo})")
        except mysql.connector.errors.IntegrityError as ie:
            conexion.rollback()
            print(f"⚠️ [AGREGAR_PERSONA] Error de integridad: {ie}")
            return JSONResponse(status_code=409, content={"mensaje": f"Error de integridad en la base de datos: {ie}"})
        finally:
            cursor.close()
            conexion.close()

        return {"mensaje": "✅ Persona agregada correctamente."}

    except Exception as e:
        print(f"❌ [AGREGAR_PERSONA] Error inesperado: {e}")
        return JSONResponse(status_code=500, content={"mensaje": f"Error inesperado al agregar persona: {e}"})
