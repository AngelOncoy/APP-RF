from fastapi import FastAPI
from app.api import face_api

app = FastAPI()

# Registrar el endpoint
app.include_router(face_api.router, prefix="/api", tags=["Face API"])

# Puedes agregar más routers después (user_api, etc.)
