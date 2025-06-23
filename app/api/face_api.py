from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import Response
import tempfile
import shutil

from app.models.user import User
from app.schemas.user_schema import (
    CompareResponse,
    UserResponse,
    UserRegisterResponse,
    UserUpdateResponse,
    UserDeleteResponse,
    UserListResponse,
    UserListItem,
    UserProfileResponse
)
from app.services.db_operations import (
    save_user_to_db,
    delete_user,
    update_user,
    get_all_users_basic,
    get_user_image,
    get_user_profile
)
from app.services.face_recognition import extract_face_features
from app.utils.image_processing import image_to_bytes

router = APIRouter()

print("face_api importado OK")

@router.get("/health")
async def health():
    print("GET /health llamado")
    return {"status": "ok"}


@router.post("/comparar", response_model=CompareResponse)
async def comparar_rostro(file: UploadFile = File(...)):
    print("POST /comparar llamado")
    try:
        import tempfile, shutil
        from app.controllers.face_controller import compare_external_image

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_image_path = temp_file.name

        # Umbral de distancia puede ajustarse dinámicamente si quieres
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

        return CompareResponse(
            match=result['match'],
            similarity=result['similarity'],
            user_data=user_resp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")


@router.post("/registrar_usuario", response_model=UserRegisterResponse)
async def registrar_usuario(
    user_id: str = Form(...),
    name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    requisitioned: bool = Form(...),
    file: UploadFile = File(...)
):
    print("POST /registrar_usuario llamado")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_image_path = temp_file.name

        image_bytes = image_to_bytes(temp_image_path)
        features = extract_face_features(temp_image_path)

        user = User(
            user_id=user_id,
            name=name,
            last_name=last_name,
            email=email,
            requisitioned=requisitioned,
            image=image_bytes,
            features=features
        )

        save_user_to_db(user)
        return UserRegisterResponse(message="Usuario registrado correctamente.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al registrar usuario: {e}")


@router.get("/listar_usuarios", response_model=UserListResponse)
async def listar_usuarios():
    print("GET /listar_usuarios llamado")
    try:
        users_db = get_all_users_basic()
        users_list = [
            UserListItem(
                user_id=u[0],
                name=u[1],
                last_name=u[2],
                email=u[3],
                requisitioned=u[4]
            ) for u in users_db
        ]
        return UserListResponse(users=users_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar usuarios: {e}")


@router.get("/usuario/{user_id}/imagen")
async def obtener_imagen_usuario(user_id: str):
    print(f"GET /usuario/{user_id}/imagen llamado")
    try:
        image_bytes = get_user_image(user_id)

        if image_bytes is None:
            raise HTTPException(status_code=404, detail="Imagen no encontrada para el usuario.")

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener imagen: {e}")


@router.put("/editar_usuario/{user_id}", response_model=UserUpdateResponse)
async def editar_usuario(
    user_id: str,
    name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    requisitioned: bool = Form(...),
    file: UploadFile = File(None)
):
    print(f"PUT /editar_usuario/{user_id} llamado")
    try:
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_image_path = temp_file.name

            image_bytes = image_to_bytes(temp_image_path)
            features = extract_face_features(temp_image_path)

            update_user(user_id, name, last_name, email, requisitioned, image_bytes, features)
        else:
            update_user(user_id, name, last_name, email, requisitioned)

        return UserUpdateResponse(message="Usuario actualizado correctamente.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar usuario: {e}")


@router.delete("/eliminar_usuario/{user_id}", response_model=UserDeleteResponse)
async def eliminar_usuario(user_id: str):
    print(f"DELETE /eliminar_usuario/{user_id} llamado")
    try:
        delete_user(user_id)
        return UserDeleteResponse(message="Usuario eliminado correctamente.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar usuario: {e}")


@router.get("/usuario/{user_id}", response_model=UserProfileResponse)
async def obtener_usuario(user_id: str):
    print(f"GET /usuario/{user_id} llamado")
    try:
        user = get_user_profile(user_id)

        if user is None:
            raise HTTPException(status_code=404, detail="Usuario no encontrado.")

        return UserProfileResponse(
            user_id=user[0],
            name=user[1],
            last_name=user[2],
            email=user[3],
            requisitioned=bool(user[4])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener usuario: {e}")
