# api/main.py - FastAPI backend integrating model and utilities
from fastapi import FastAPI, File, UploadFile, BackgroundTasks # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from model.predict import predict_image
from model.predict_video import predict_video_file
from model.train import train_model
from utils.realtime_batch import process_webcam_stream, process_folder
import shutil
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Deepfake Detection API is running."}

from utils.dataset_loader import download_and_prepare

@app.on_event("startup")
def fetch_data():
    download_and_prepare()
    print("[INFO] Dataset downloaded and prepared.")
    print("[INFO] All tasks completed successfully.")

@app.post("/predict/image")
def predict_image_route(file: UploadFile = File(...)):
    try:
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        label, confidence = predict_image(temp_path)
        return JSONResponse({"label": label, "confidence": float(confidence)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict/video")
def predict_video_route(file: UploadFile = File(...)):
    try:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = predict_video_file(temp_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/scan/webcam")
def scan_webcam():
    try:
        result = process_webcam_stream()
        return JSONResponse({"message": "Webcam scan complete", "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/scan/folder")
def scan_folder():
    try:
        result = process_folder("data")
        return JSONResponse({"message": "Folder scan complete", "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(train_model, data_dir="data", log_dir="logs/tensorboard")
        return JSONResponse({"message": "Training started in background."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
