# model/predict_video.py - Frame-based video prediction logic
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

MODEL_PATH = "saved_model/deepfake_cnn.h5"
IMAGE_SIZE = (128, 128)

model = load_model(MODEL_PATH)

def predict_video(video_path: str, threshold: float = 0.5, frame_skip: int = 10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_resized = cv2.resize(frame, IMAGE_SIZE)
            frame_array = img_to_array(frame_resized) / 255.0
            frame_array = np.expand_dims(frame_array, axis=0)

            pred = model.predict(frame_array)[0][0]
            predictions.append(pred)
        frame_count += 1

    cap.release()

    if not predictions:
        raise ValueError("No frames were processed. Check frame_skip or video content.")

    avg_confidence = float(np.mean(predictions))
    label = "Fake" if avg_confidence >= threshold else "Real"

    return {
        "label": label,
        "confidence": avg_confidence,
        "frames_evaluated": len(predictions)
    }
