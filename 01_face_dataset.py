import cv2
import os

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if cascade file loaded properly
if face_detector.empty():
    print("Error: Haar cascade file not loaded. Check path!")
    exit()

# Get user name
name = input("\nEnter name of the person: ")
face_id = input(f"Enter a unique ID for {name}: ")

# Create dataset directory if it doesn't exist
dataset_path = "C:/Users/Jina/OneDrive/Desktop/face_detection/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0  # Initialize image count

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    img = cv2.flip(img, 1)  # Flip video image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    print(f"Faces detected: {len(faces)}")  # Debug message

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured face into the dataset folder with name and ID
        image_path = f"{dataset_path}/User.{face_id}.{name}.{count}.jpg"
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.equalizeHist(face_img)  # Improve contrast
        cv2.imwrite(image_path, face_img)
        print(f"Saved image: {image_path}")  # Debug message

        cv2.imshow("image", img)

    # Press 'ESC' to exit or save 100 images and stop
    k = cv2.waitKey(100) & 0xFF
    if k == 27 or count >= 100:  # Increase image count
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
