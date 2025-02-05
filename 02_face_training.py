import os
import cv2
import numpy as np
from PIL import Image
import pickle  # Ensure you import pickle

# Path to the dataset
dataset_path = "dataset"

# Prepare the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare lists to store face images and their corresponding labels
faces = []
labels = []

# Prepare the names list to map labels to person names
names = {}

# Traverse through the dataset and process images
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Add support for .png files
            image_path = os.path.join(root, file)
            # Extract the label from the filename, assuming format: User.name.X.jpg
            label_name = file.split('.')[0].split('User.')[-1]
            # Get the person's label number from the filename
            label = int(label_name) if label_name.isdigit() else -1  # Safeguard for invalid filenames

            # Ensure the filename is valid for labeling
            if label != -1:
                # Convert image to grayscale
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Use the Haar Cascade Classifier for face detection
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces_detected:
                    face_img = gray[y:y + h, x:x + w]
                    faces.append(face_img)
                    labels.append(label)

                    # Store the name associated with the label (name is part of the filename)
                    names[label] = label_name

# Check if faces were detected
if len(faces) == 0:
    print("No faces detected!")
else:
    # Train the recognizer with the face images and labels
    recognizer.train(faces, np.array(labels))

    # Save the trained model to a file
    recognizer.save("trainer.yml")

    # Save the names dictionary (name-label mapping) to a file
    with open("names.pickle", 'wb') as f:
        pickle.dump(names, f)

    print("Training completed and model saved.")
