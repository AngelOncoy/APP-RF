from fastapi import APIRouter, UploadFile, File, HTTPException
from app.controllers.face_controller import compare_external_image
from app.schemas.user_schema import CompareResponse, UserResponse
import tempfile
import shutil

router = APIRouter()

@router.post("/comparar", response_model=CompareResponse)
async def comparar_rostro(file: UploadFile = File(...)):
    """
    Endpoint para comparar una imagen facial contra la base de datos.
    """
    try:
        # Guardar temporalmente la imagen recibida
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_image_path = temp_file.name

        # Llamar al comparador
        result = compare_external_image(temp_image_path)

        if result['match']:
            user = result['user_data']
            user_resp = UserResponse(
                user_id=user['user_id'],
                name=user['name'],
                last_name=user['last_name'],
                email=user['email'],
                requisitioned=user['requisitioned']
            )
        else:
            user_resp = None

        # Preparar la respuesta
        response = CompareResponse(
            match=result['match'],
            similarity=result['similarity'],
            user_data=user_resp
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")
