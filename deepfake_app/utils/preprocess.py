# utils/preprocess.py - Image preprocessing utilities
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

def preprocess_image(img_path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Load, resize, normalize, and convert image to array for CNN input.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_frame(frame, target_size=(224, 224)) -> np.ndarray:
    """
    Resize, normalize, and format webcam or video frame
    """
    if frame is None:
        raise ValueError("Empty frame received")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float32") / 255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame
