from pydantic import BaseModel

class PersonaRespuesta(BaseModel):
    id: int
    nombre: str
    apellido: str
    correo: str
    foto: str
    similitud: float
