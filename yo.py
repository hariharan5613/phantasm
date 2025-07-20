import os
import datetime
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path

from ultralytics import YOLO  # Ensure you have ultralytics installed
from config import INTRUDER_DIR, MODEL_PATH, LABELS_PATH, TRAIN_DIR, USER_PINS

# Setup
os.makedirs(INTRUDER_DIR, exist_ok=True)
root = tk.Tk()
root.withdraw()

# Load or train face recognition model
if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recogniser = cv2.face.LBPHFaceRecognizer_create()

    faces, labels = [], []
    label_dict = {}
    current_label = 0

    for person_dir in Path(TRAIN_DIR).iterdir():
        if not person_dir.is_dir():
            continue
        label_dict[current_label] = person_dir.name
        for img_path in person_dir.glob("*.[jp][pn]g"):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
                faces.append(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
                labels.append(current_label)
                break
        current_label += 1

    recogniser.train(np.array(faces), np.array(labels))
    recogniser.write(MODEL_PATH)
    np.save(LABELS_PATH, label_dict)
else:
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.read(MODEL_PATH)
    label_dict = np.load(LABELS_PATH, allow_pickle=True).item()

# Load YOLOv5 model
yolo_model = YOLO("yolov5m.pt")  # Ensure the model is downloaded

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_detected = False
    results = yolo_model(frame)[0]  # YOLOv5 output format

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if yolo_model.names[cls] != "person":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (200, 200))

        try:
            label_id, confidence = recogniser.predict(resized)
            name = label_dict.get(label_id, "Unknown")
        except Exception:
            continue

        if confidence < 80:  # Confidence threshold
            status, color = f"{name}: Access Granted", (0, 255, 0)
        else:
            pin = simpledialog.askstring("PIN Required", f"{name}, face not matched. Enter PIN:", show='*')
            if pin == USER_PINS.get(name):
                status, color = f"{name}: PIN Verified", (0, 255, 0)
            else:
                status, color = f"{name}: Access Denied", (0, 0, 255)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(INTRUDER_DIR, f"intruder_{ts}.jpg"), frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        face_detected = True
        break  # Only process one face

    if not face_detected:
        cv2.putText(frame, "No recognized face", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
