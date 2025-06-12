from fastapi import APIRouter,Form, UploadFile, File, HTTPException
from app.controllers.face_controller import compare_external_image
from app.models.user import User
from app.schemas.user_schema import CompareResponse, UserResponse, UserRegisterResponse
import tempfile
import shutil

from app.services.db_operations import save_user_to_db
from app.services.face_recognition import extract_face_features
from app.utils.image_processing import image_to_bytes

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

@router.post("/registrar_usuario", response_model=UserRegisterResponse)
async def registrar_usuario(
    user_id: str = Form(...),
    name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    requisitioned: bool = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint para registrar un nuevo usuario con imagen.
    """
    try:
        # Guardar temporalmente la imagen
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_image_path = temp_file.name

        # Procesar imagen
        image_bytes = image_to_bytes(temp_image_path)
        features = extract_face_features(temp_image_path)

        # Crear objeto User
        user = User(
            user_id=user_id,
            name=name,
            last_name=last_name,
            email=email,
            requisitioned=requisitioned,
            image=image_bytes,
            features=features
        )

        # Guardar en DB
        save_user_to_db(user)

        return UserRegisterResponse(message="Usuario registrado correctamente.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al registrar usuario: {e}")


from app.services.db_operations import get_all_users_basic
from app.schemas.user_schema import UserListResponse, UserListItem


@router.get("/listar_usuarios", response_model=UserListResponse)
async def listar_usuarios():
    """
    Endpoint para listar todos los usuarios.
    """
    try:
        users_db = get_all_users_basic()

        users_list = [
            UserListItem(
                user_id=u[0],
                name=u[1],
                last_name=u[2],
                email=u[3],
                requisitioned=u[4]
            )
            for u in users_db
        ]

        return UserListResponse(users=users_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar usuarios: {e}")
