from fastapi import FastAPI
from .endpoints import router

app = FastAPI(
    title="Reconocimiento Facial Offline",
    description="API local para reconocimiento facial sin internet",
    version="1.0"
)

# Agrega las rutas
app.include_router(router)
