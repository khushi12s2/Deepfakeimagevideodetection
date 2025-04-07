from fastapi import APIRouter, HTTPException, Depends, status, FastAPI, File, UploadFile, BackgroundTasks # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # type: ignore
from pydantic import BaseModel
from datetime import datetime, timedelta
import hashlib
from jose import JWTError, jwt # type: ignore
from typing import Optional
import streamlit as st
import requests

SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

BASE_URL = "http://localhost:8000"

users_db = {}
refresh_tokens_db = {}

auth = APIRouter()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class User(BaseModel):
    username: str
    password: str
    role: str = "user"

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: str
    role: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{BASE_URL}/auth/token")

@auth.post("/register")
def register(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    users_db[user.username] = {
        "password": hash_password(user.password),
        "role": user.role
    }
    return {"msg": "User registered successfully"}

@auth.post("/auth/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    user_record = users_db.get(username)
    if not user_record or user_record["password"] != hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    role = user_record["role"]
    access_token = create_access_token({"sub": username, "role": role})
    refresh_token = create_refresh_token({"sub": username, "role": role})
    refresh_tokens_db[username] = refresh_token
    return Token(access_token=access_token, refresh_token=refresh_token)

def create_access_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@auth.post("/auth/refresh", response_model=Token)
def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or refresh_tokens_db.get(username) != refresh_token:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        role = payload.get("role")
        new_access_token = create_access_token({"sub": username, "role": role})
        new_refresh_token = create_refresh_token({"sub": username, "role": role})
        refresh_tokens_db[username] = new_refresh_token
        return Token(access_token=new_access_token, refresh_token=new_refresh_token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return TokenData(username=username, role=role)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(required_role: str):
    def role_checker(user: TokenData = Depends(get_current_user)):
        if user.role != required_role:
            raise HTTPException(status_code=403, detail="Access denied: insufficient permissions")
        return user
    return role_checker

@auth.get("/users/me")
def read_users_me(user: TokenData = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}

@auth.get("/admin/dashboard")
def admin_dashboard(user: TokenData = Depends(require_role("admin"))):
    return {"msg": f"Welcome Admin {user.username}!"}

@auth.get("/moderator/review")
def moderator_review(user: TokenData = Depends(require_role("moderator"))):
    return {"msg": f"Moderator {user.username}, ready to review submissions."}

# Example route placeholders to wire model pipeline
@auth.post("/model/train")
def train_model(background_tasks: BackgroundTasks, user: TokenData = Depends(require_role("admin"))):
    background_tasks.add_task(dummy_train_model)
    return {"msg": "Model training initiated."}

def dummy_train_model():
    import time
    time.sleep(3)  # Simulate model training
    print("Model trained")

@auth.post("/model/predict")
def predict_image(file: UploadFile = File(...), user: TokenData = Depends(get_current_user)):
    return {"filename": file.filename, "result": "fake", "confidence": 0.92}

@auth.get("/logs/sessions")
def get_user_logs(user: TokenData = Depends(require_role("admin"))):
    return {"logs": [f"User {user.username} accessed logs."]}
