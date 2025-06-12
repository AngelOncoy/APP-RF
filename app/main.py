from fastapi import FastAPI
from app.api import face_api

app = FastAPI(
    title="Reconocimiento Facial API con HOG",
    version="2.0",
    description="API para reconocimiento facial"

)

# Registrar el endpoint
app.include_router(face_api.router, prefix="/api", tags=["Face API"])

# Puedes agregar más routers después (user_api, etc.)

