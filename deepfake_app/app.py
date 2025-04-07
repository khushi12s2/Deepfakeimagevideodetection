# app.py - Streamlit frontend connected to FastAPI backend with advanced features
import streamlit as st
import requests
from PIL import Image
import io
import time
import matplotlib.pyplot as plt # type: ignore
import pandas as pd

st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🧠 Deepfake Image & Video Detector")

backend_url = "http://localhost:8000"

tabs = st.tabs(["📸 Image", "🎥 Video", "🔁 Train", "📂 Folder Scan", "📊 Logs"])

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    polling_interval = st.slider("Training Status Poll Interval (sec)", 1, 30, 5)
    auth_token = st.text_input("🔐 API Token", type="password", value="your-secret-token")
    headers = {"Authorization": f"Bearer {auth_token}"}

with tabs[0]:
    st.subheader("🔍 Upload Image for Detection")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        files = {"file": uploaded_img.getvalue()}
        response = requests.post(f"{backend_url}/predict/image", files=files, headers=headers)
        result = response.json()
        if result.get("confidence", 0) >= threshold:
            st.success(f"Label: {result['label']} | Confidence: {result['confidence']:.2f}")
        else:
            st.warning("Below threshold. Possibly not fake.")
        st.download_button("Download Result", str(result), file_name="image_prediction.json")

with tabs[1]:
    st.subheader("🎥 Upload Video for Detection")
    uploaded_vid = st.file_uploader("Choose a video", type=["mp4"])
    if uploaded_vid:
        files = {"file": uploaded_vid.getvalue()}
        response = requests.post(f"{backend_url}/predict/video", files=files, headers=headers)
        result = response.json()
        st.json(result)
        st.download_button("Download Video Report", str(result), file_name="video_prediction.json")

with tabs[2]:
    st.subheader("🧠 Train Model")
    if st.button("🔁 Start Model Training"):
        response = requests.post(f"{backend_url}/train", headers=headers)
        st.success(response.json().get("message", "Training started."))
        with st.spinner("Waiting for training to complete..."):
            for _ in range(20):
                time.sleep(polling_interval)
                st.info("Training is still in progress... (Polling placeholder)")
            st.success("Training likely completed. Check TensorBoard for details.")

        st.markdown("### 📈 Accuracy & Loss Visualization")
        try:
            hist_df = pd.read_csv("logs/training_history.csv")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(hist_df["accuracy"], label="Train Acc")
            ax[0].plot(hist_df["val_accuracy"], label="Val Acc")
            ax[0].set_title("Model Accuracy")
            ax[0].legend()
            ax[1].plot(hist_df["loss"], label="Train Loss")
            ax[1].plot(hist_df["val_loss"], label="Val Loss")
            ax[1].set_title("Model Loss")
            ax[1].legend()
            st.pyplot(fig)

            # Download buttons
            st.download_button("Download Training History CSV", hist_df.to_csv(index=False), file_name="training_history.csv")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("Download Accuracy & Loss Chart", buf.getvalue(), file_name="training_plot.png", mime="image/png")
        except Exception as e:
            st.warning("Training history not found. Run training first.")

        st.markdown("### 🔬 TensorBoard Log Viewer")
        st.components.v1.iframe("http://localhost:6006", height=600, scrolling=True)

with tabs[3]:
    st.subheader("📂 Folder & Webcam Scan")
    if st.button("📸 Realtime Webcam Scan"):
        response = requests.get(f"{backend_url}/scan/webcam", headers=headers)
        st.json(response.json())
    if st.button("📂 Scan Folder"):
        response = requests.get(f"{backend_url}/scan/folder", headers=headers)
        st.json(response.json())

with tabs[4]:
    st.subheader("📊 Session Logs & History")
    try:
        df = pd.read_csv("logs/session_log.csv")
        st.dataframe(df)
        st.download_button("Download Log CSV", df.to_csv(index=False), file_name="session_log.csv")
    except Exception as e:
        st.warning("No logs found. Upload and predict first to generate logs.")

from utils.dataset_loader import download_and_prepare

if st.sidebar.button("📥 Load Kaggle Dataset"):
    with st.spinner("Downloading and organizing data..."):
        download_and_prepare()
    st.success("Dataset downloaded and sorted into `data/real` and `data/fake`.")
    st.markdown("### 📂 Dataset Structure")
    st.write("The dataset is organized as follows:")
    st.write("```\n"
             "data/\n"
             "├── fake/\n"
             "│   ├── fake1.jpg\n"
             "│   ├── fake2.jpg\n"
             "│   └── ...\n"
             "└── real/\n"
             "    ├── real1.jpg\n"
             "    ├── real2.jpg\n"
             "    └── ...\n"
             "```")
    print("[INFO] Dataset downloaded and prepared.")
    print("[INFO] All tasks completed successfully.")       