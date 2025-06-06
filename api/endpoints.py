from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from .schema import PersonaRespuesta
from Reconocimiento_Facial import buscar_persona_por_imagen

router = APIRouter()

# --- Endpoint /comparar ---
@router.post("/comparar", response_model=PersonaRespuesta)
async def comparar_imagen(file: UploadFile = File(...)):
    try:
        # Guardar la imagen temporalmente
        image_bytes = await file.read()
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)

        # Llamar a la función que ya tienes implementada
        resultado, mensaje = buscar_persona_por_imagen(temp_image_path, umbral_distancia=0.75)

        # Borrar la imagen temporal
        os.remove(temp_image_path)

        # Si no hubo coincidencia
        if resultado is None:
            raise HTTPException(status_code=404, detail=mensaje)

        # Si hubo coincidencia, construir respuesta
        return {
            "id": resultado['id'],
            "nombre": resultado['nombre'],
            "apellido": resultado['apellido'],
            "correo": resultado['correo'],
            "foto": resultado['foto'],
            "similitud": mensaje * 100  # convertir a porcentaje
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
