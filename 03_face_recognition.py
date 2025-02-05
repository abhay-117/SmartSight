import cv2
import numpy as np
import os
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = 'C:/Users/Jina/OneDrive/Desktop/face_detection/trainer/trainer.yml'

if not os.path.exists(trainer_path):
    print("[ERROR] Model not found! Train the model first.")
    exit()

recognizer.read(trainer_path)
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadePath)

# Load name mappings
with open('names.pkl', 'rb') as f:
    names = pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize id counter
id = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size for face recognition
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))  # Normalize size
        id, confidence = recognizer.predict(face_img)

        if confidence < 70:  # Increase threshold for better match
            name = names.get(id, "Unknown")
        else:
            name = "Unknown"

        confidence_text = f"{round(100 - confidence)}%"
        cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h + 20), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xFF  # Press 'ESC' to exit
    if k == 27:
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
