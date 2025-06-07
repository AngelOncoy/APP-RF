#main.py
from fastapi import FastAPI
from api import endpoints

app = FastAPI(
    title="Reconocimiento Facial API",
    description="API para reconocimiento facial",
    version="1.0"
)

# Agrega las rutas
app.include_router(endpoints.router)
