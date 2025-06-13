import os
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Reconocimiento Facial API con HOG",
    version="2.0",
    description="API para reconocimiento facial"
)

# Probar si la import de face_api es la que rompe:
try:
    from app.api import face_api
    app.include_router(face_api.router, prefix="/api", tags=["Face API"])
    print("face_api importado OK")
except Exception as e:
    print(f"ERROR importando face_api: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
