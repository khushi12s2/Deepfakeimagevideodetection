# model/predict.py - Image inference logic
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

MODEL_PATH = "saved_model/deepfake_cnn.h5"
IMAGE_SIZE = (128, 128)

model = load_model(MODEL_PATH)

def predict_image(img_path: str, threshold: float = 0.5):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path does not exist: {img_path}")

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Fake" if prediction >= threshold else "Real"

    return {
        "label": label,
        "confidence": float(prediction)
    }