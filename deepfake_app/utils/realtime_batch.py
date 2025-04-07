# utils/realtime_batch.py - Webcam and Folder Scanning
import cv2
import argparse
import os
import time
from model.predict import predict_image_array
from utils.preprocess import preprocess_frame

def scan_webcam(threshold=0.5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("[INFO] Press 'q' to exit webcam detection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        preprocessed = preprocess_frame(frame)
        label, confidence = predict_image_array(preprocessed)

        text = f"{label.upper()} ({confidence:.2f})"
        color = (0, 255, 0) if label == "real" else (0, 0, 255)
        if confidence >= threshold:
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("DeepFake Webcam Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def scan_folder(folder_path, threshold=0.5):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        return

    print(f"[INFO] Scanning folder: {folder_path}")
    image_ext = (".jpg", ".jpeg", ".png")
    for file in os.listdir(folder_path):
        if file.lower().endswith(image_ext):
            full_path = os.path.join(folder_path, file)
            try:
                preprocessed = preprocess_frame(cv2.imread(full_path))
                label, confidence = predict_image_array(preprocessed)
                print(f"{file}: {label.upper()} ({confidence:.2f})")
            except Exception as e:
                print(f"Failed to process {file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time webcam or folder batch scanning")
    parser.add_argument("--webcam", action="store_true", help="Enable webcam mode")
    parser.add_argument("--folder", type=str, help="Scan folder with images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    args = parser.parse_args()

    if args.webcam:
        scan_webcam(threshold=args.threshold)
    elif args.folder:
        scan_folder(args.folder, threshold=args.threshold)
    else:
        print("Please specify --webcam or --folder <path>")
        print("Use --help for more information.")
# utils/realtime_batch.py - Webcam and Folder Scanning      