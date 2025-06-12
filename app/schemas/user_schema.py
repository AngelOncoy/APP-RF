from pydantic import BaseModel
from typing import Optional
from typing import List


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

class UserListItem(BaseModel):
    user_id: str
    name: str
    last_name: str
    email: str
    requisitioned: bool

class UserListResponse(BaseModel):
    users: List[UserListItem]

class UserUpdateResponse(BaseModel):
    message: str

class UserDeleteResponse(BaseModel):
    message: str

class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    last_name: str
    email: str
    requisitioned: bool
