from pydantic import BaseModel
from typing import Optional

class UserResponse(BaseModel):
    user_id: str
    name: str
    last_name: str
    email: str
    requisitioned: bool

class CompareResponse(BaseModel):
    match: bool
    similarity: float
    user_data: Optional[UserResponse]

class UserRegisterResponse(BaseModel):
    message: str