import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # For local development only
import cv2
import numpy as np
import threading
import base64
import requests
import pandas as pd
import pytesseract
import pyttsx3
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, send_file
from flask_cors import CORS, cross_origin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask_dance.contrib.google import make_google_blueprint, google
from functools import wraps

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

# --- Application Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24)  # More secure secret key
CORS(app)



# --- Context Processor for Templates ---
@app.context_processor
def inject_user():
    return dict(
        is_authenticated=('user' in session),
        current_user=session.get('user', None)
    )

# --- Directories and Paths ---
for folder in ['static', 'face_training', 'data', 'recognizer']:
    if not os.path.exists(folder):
        os.makedirs(folder)

MODEL_PATH = os.path.join('recognizer', 'TrainingData.yml')
LABEL_MAP_PATH = os.path.join('recognizer', 'label_dict.npy')

# --- LBPH Recognizer Initialization ---
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=10, grid_x=8, grid_y=8, threshold=70)
label_map = {}  # This will be a dict mapping person name to label (int)

if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    print("✅ Model and label map loaded successfully.")
else:
    print("⚠️ No existing model found. Please add training images and trigger training.")

# --- YOLOv8 Model Initialization ---
yolo_model = YOLO(r"C:\Users\Jina\OneDrive\Desktop\ml_flask\yolov8x.pt")

# --- Google OAuth Setup via Flask-Dance ---
# Change the redirect_url to a new endpoint to avoid duplicate endpoint errors.
google_bp = make_google_blueprint(
    client_id="793570230450-hqr85kj115tqsj1lsoosab48d70k96j0.apps.googleusercontent.com",
    client_secret="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",
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ],
    redirect_url="/login/google/process"  # Use a new URL for custom processing
)
app.register_blueprint(google_bp, url_prefix="/login")

# --- In-Memory User Store and Excel File ---
users = {
    "admin": "password",
    "bobby": "123"
}
USER_FILE_PATH = r"C:\Users\Jina\OneDrive\Desktop\ml_flask\users.xlsx"

def read_users():
    try:
        df = pd.read_excel(USER_FILE_PATH)
        users_dict = {}
        for _, row in df.iterrows():
            users_dict[row['username']] = {
                "password": row['password'],  # In production, store hashed passwords!
                "emergency_contact": row.get('emergency_contact', ''),
                "email": row.get('email', '')
            }
        return users_dict
    except FileNotFoundError:
        return {}

def write_users(users_dict):
    data = []
    for username, info in users_dict.items():
        data.append({
            "username": username,
            "password": info.get("password"),
            "emergency_contact": info.get("emergency_contact", ""),
            "email": info.get("email", "")
        })
    df = pd.DataFrame(data)
    # Remove existing file (if any) to avoid permission or locking issues
    if os.path.exists(USER_FILE_PATH):
        try:
            os.remove(USER_FILE_PATH)
        except Exception as e:
            print(f"Error removing existing file: {e}")
    with pd.ExcelWriter(USER_FILE_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)


# --- Custom Google Process Route ---
@app.route('/login/google/process')
@cross_origin()
def google_process():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google.", "danger")
        return redirect(url_for("signin"))
    user_info = resp.json()
    email = user_info.get("email", "Unknown")
    name = user_info.get("name", "")
    
    users_db = read_users()
    # Convert the stored email to string to avoid errors if it is NaN
    existing_user = next((uname for uname, udata in users_db.items() 
                          if str(udata.get("email", "")).lower() == email.lower()), None)
    if existing_user:
        # User exists; log them in
        session['user'] = existing_user
        flash("Logged in with Google successfully!", "success")
        return redirect(url_for('home'))
    else:
        # Save Google details in session for pre-filling the signup form.
        session['google_email'] = email
        session['google_name'] = name
        flash("No account found. Please sign up to complete registration.", "warning")
        return redirect(url_for('signup'))


# --- Camera and TTS State Management ---
face_cap = None
tts_cap = None
camera_active = False
camera_lock = threading.Lock()
camera_active = False
camera_lock = threading.Lock()
cap = None
tts_lock = threading.Lock()

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1)

# --- Image Preprocessing for OCR ---
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    alpha = 1.5
    beta = 10
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return gray

def extract_text_from_image(image):
    try:
        processed_image = preprocess_image(image)
        extracted_text = pytesseract.image_to_string(processed_image, lang="eng")
        return extracted_text.strip()
    except Exception as e:
        print(f"Error using Tesseract OCR: {e}")
        return ""

# --- Text-to-Speech using pyttsx3 ---
def text_to_speech_pyttsx3(text):
    with tts_lock:
        try:
            audio_file = os.path.join('static', "tts_output.wav")
            tts_engine.save_to_file(text, audio_file)
            tts_engine.runAndWait()
            return audio_file
        except Exception as e:
            print(f"Error during TTS: {e}")
            return None

# --- Capture Image from Camera ---
def capture_frame(cap):
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
    ret, frame = cap.read()
    if ret:
        return frame
    return None


# --- Face Detection using YOLOv8 ---
def detect_faces(image):
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    for i, box in enumerate(boxes):
        if confidences[i] < 0.5:
            continue
        x1, y1, x2, y2 = box.astype("int")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Face", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# --- Detect and Crop Face for Training ---
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    face_crop = cv2.equalizeHist(face_crop)
    face_crop = cv2.resize(face_crop, (200, 200))
    print("✅ Face detected and cropped successfully.")
    return face_crop

# --- Object Detection with YOLOv8 ---
def determine_avoidance_path(detected_objects, frame_width):
    left_blocked = False
    right_blocked = False
    center_blocked = False
    for obj, position in detected_objects.items():
        if position == "on the left":
            left_blocked = True
        elif position == "on the right":
            right_blocked = True
        elif position == "in the center":
            center_blocked = True
    if center_blocked and not left_blocked and not right_blocked:
        return "both left and right are usable"
    if center_blocked:
        if not left_blocked:
            return "left"
        elif not right_blocked:
            return "right"
        else:
            return "back"
    if left_blocked and not right_blocked:
        return "right"
    if right_blocked and not left_blocked:
        return "left"
    return "safe"

def detect_objects(image):
    (H, W) = image.shape[:2]
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    classIDs = results[0].boxes.cls.cpu().numpy()
    detected_objects = {}
    for i, box in enumerate(boxes):
        if confidences[i] < 0.5:
            continue
        x1, y1, x2, y2 = box.astype("int")
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w/2
        if center_x < W/3:
            pos = "on the left"
        elif center_x < 2 * W/3:
            pos = "in the center"
        else:
            pos = "on the right"
        label = yolo_model.names[int(classIDs[i])]
        detected_objects[label] = pos
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = "{}: {:.2f} ({})".format(label, confidences[i], pos)
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    avoidance_path = determine_avoidance_path(detected_objects, W)
    cv2.putText(image, "Avoidance: " + avoidance_path, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    return image

# --- Video Frame Generator (Local Camera) ---
def generate_frames():
    global cap, camera_active
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            try:
                label_id, distance = recognizer.predict(face)
                confidence_percent = int(100 * (1 - distance/300)) if distance < 300 else 0
                if confidence_percent > 70:
                    name = [k for k, v in label_map.items() if v == label_id]
                    name = name[0] if name else "Unknown"
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
            except Exception as e:
                print(f"Face recognition error: {e}")
                name = "Error"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{name} ({confidence_percent}%)', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
def generate_frames_for_text_to_speech():
    global tts_cap
    if tts_cap is None or not tts_cap.isOpened():
        tts_cap = cv2.VideoCapture(0)
    while True:
        success, frame = tts_cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
# --- Live Object Detection Video Frame Generator ---
def generate_object_frames():
    global cap, camera_active
    cap = cv2.VideoCapture(0)
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detect_objects(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# --- ESP32 Video Frame Generator ---
import time  # added at the top with the other imports
def generate_esp32_frames(url):
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print(f"ESP32 returned status code: {response.status_code}")
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching ESP32 frame: {e}")
            time.sleep(0.1)
            continue

# --- Flask Routes ---

@app.route('/')
@cross_origin()
def landing():
    if 'user' in session:
        return redirect(url_for('home'))
    return render_template('landing.html')

@app.route('/home')
@login_required
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/signin', methods=['GET', 'POST'])
@cross_origin()
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users_db = read_users()
        if username in users_db and users_db[username]["password"] == password:
            session['user'] = username
            flash("Signed in successfully!", "success")
            return redirect(url_for('home'))
        flash("Invalid credentials!", "danger")
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
@cross_origin()
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        emergency_contact = request.form.get('emergency_contact', '').strip()
        email = request.form.get('email', '').strip()
        if not all([username, password, confirm_password, emergency_contact, email]):
            flash("Please fill out all fields.", "danger")
            return render_template('signup.html')
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return render_template('signup.html')
        users_db = read_users()
        if any(u.lower() == username.lower() or (users_db[u].get('email', '').lower() == email.lower()) for u in users_db):
            flash("Username or email already exists!", "danger")
            return render_template('signup.html')
        users_db[username] = {
            "password": password,
            "emergency_contact": emergency_contact,
            "email": email
        }
        write_users(users_db)
        flash("Account created successfully!", "success")
        session['user'] = username  # Auto log in the user after signup
        return redirect(url_for('home'))
    return render_template('signup.html', 
                           google_name=session.get('google_name', ""), 
                           google_email=session.get('google_email', ""))

@app.route('/logout')
@cross_origin()
def logout():
    session.pop('user', None)
    session.pop('google_email', None)
    session.pop('google_name', None)
    flash("Logged out successfully!", "success")
    return redirect(url_for('landing'))

@app.route('/video_feed')
@cross_origin()
def video_feed_tts():
    global camera_active
    with camera_lock:
        camera_active = True
    return Response(generate_frames_for_text_to_speech(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_training')
@cross_origin()
def video_feed_training():
    global camera_active_training, cap_training
    with camera_lock:
        if not camera_active_training:
            cap_training = cv2.VideoCapture(0)
            camera_active_training = True

        while camera_active_training and cap_training.isOpened():
            success, frame = cap_training.read()
            if not success:
                break
            
            # Display the frame using OpenCV window
            cv2.imshow('Training Feed', frame)
            
            # Close window if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the window
        camera_active_training = False
        cap_training.release()
        cv2.destroyAllWindows()

    return jsonify({"message": "Training camera stopped."})


@app.route('/video_feed_face')
@cross_origin()
def video_feed_face():
    global camera_active
    with camera_lock:
        camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_object_detection')
@cross_origin()
def video_feed_object_detection():
    global camera_active
    with camera_lock:
        camera_active = True
    return Response(generate_object_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_esp32')
@cross_origin()
def video_feed_esp32():
    esp_url = request.args.get('url')
    if not esp_url:
        return "ESP32 URL not provided", 400
    return Response(generate_esp32_frames(esp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
@cross_origin()
def stop_camera():
    global camera_active, cap
    with camera_lock:
        if camera_active:
            camera_active = False
            if cap is not None:
                cap.release()
                cap = None  # Reset cap to avoid conflicts
    return jsonify({"message": "Camera stopped"})


@app.route('/save_training', methods=['POST'])
@cross_origin()
def save_training():
    data = request.get_json()
    if not data or 'name' not in data or 'image' not in data or 'count' not in data:
        return jsonify({"status": "error", "message": "Invalid data"}), 400
    trainee_name = data['name'].strip()
    image_data = data['image']
    count = data['count']
    max_training_images = 100
    trainee_folder = os.path.join('data', trainee_name)
    os.makedirs(trainee_folder, exist_ok=True)
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        print(f"⚠️ Base64 decoding error: {e}")
        return jsonify({"status": "error", "message": f"Decoding error: {e}"}), 400
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("⚠️ Could not decode image.")
        return jsonify({"status": "error", "message": "Could not decode image"}), 400
    face_crop = detect_and_crop_face(img)
    if face_crop is None:
        return jsonify({"status": "error", "message": "No face detected. Try better lighting or face positioning."}), 400
    filename = os.path.join(trainee_folder, f"{count}.png")
    cv2.imwrite(filename, face_crop)
    if count >= max_training_images:
        with camera_lock:
            global camera_active, cap
            camera_active = False
            if cap is not None:
                cap.release()
                cap = None
                print("🛑 Camera stopped after reaching max training images.")
    threading.Thread(target=train_faces).start()
    return jsonify({"status": "success", "message": f"Saved image {count}"}), 200

def train_faces():
    print("🔄 Training model with new data... [QOC Update: Starting training]")
    data_dir = 'data'
    faces = []
    labels = []
    new_label_map = {}
    current_label = 0
    if not os.path.exists(data_dir):
        print("⚠️ Data directory does not exist.")
        return
    for person_name in sorted(os.listdir(data_dir)):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name not in new_label_map:
            new_label_map[person_name] = current_label
            current_label += 1
        label = new_label_map[person_name]
        for file in sorted(os.listdir(person_path)):
            if file.endswith('.png'):
                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    faces.append(img)
                    labels.append(label)
    if len(faces) == 0:
        print("⚠️ No training data found.")
        return
    try:
        new_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=10, grid_x=8, grid_y=8, threshold=70)
        new_recognizer.train(faces, np.array(labels))
        new_recognizer.save(MODEL_PATH)
        print("✅ Training complete. Model saved as 'TrainingData.yml'. [QOC Update: Training finished]")
        np.save(LABEL_MAP_PATH, new_label_map)
        print(f"✅ Label map saved: {new_label_map}")
        global recognizer, label_map
        recognizer = new_recognizer
        label_map = new_label_map
    except Exception as e:
        print(f"❌ Training error: {e}")

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    recognized_names = []
    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        try:
            label_id, distance = recognizer.predict(face)
            confidence_percent = int(100 * (1 - distance/300)) if distance < 300 else 0
            print(f"Predicted label: {label_id}, Distance: {distance}, Confidence: {confidence_percent}%")
            if confidence_percent > 70:
                name = [k for k, v in label_map.items() if v == label_id]
                name = name[0] if name else "Unknown"
                recognized_names.append(name)
                threading.Thread(target=tts_engine.say, args=(name,)).start()
                threading.Thread(target=tts_engine.runAndWait).start()
            else:
                recognized_names.append("Unknown")
        except Exception as e:
            print(f"Recognition error: {e}")
            recognized_names.append("Recognition Error")
    return recognized_names if recognized_names else ["No Face Detected"]

@app.route('/recognize', methods=['POST'])
@cross_origin()
def recognize():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "Invalid data"}), 400
    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Could not decode image"}), 400
        result = recognize_face(img)
        return jsonify({"status": "success", "message": result})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Processing error: {e}"}), 500

camera_lock = threading.Lock()
face_detection_active = False

@app.route('/text-to-speech', methods=['GET', 'POST'])
@cross_origin()
def text_to_speech():
    global tts_cap, camera_active
    if request.method == 'POST':
        with camera_lock:
            if camera_active:
                flash("Camera is already in use for another operation.", "danger")
                return render_template('tts.html')

            camera_active = True
            frame = capture_frame(tts_cap)
            if frame is not None:
                text = extract_text_from_image(frame)
                if text:
                    audio_file = text_to_speech_pyttsx3(text)
                    if audio_file:
                        audio_url = url_for('static', filename='tts_output.wav')
                        return render_template('tts.html', audio_url=audio_url, extracted_text=text)
                else:
                    flash("No text found in the image.", "danger")

            if tts_cap is not None:
                tts_cap.release()
                tts_cap = None

            camera_active = False

    return render_template('tts.html')



@app.route('/face-detection', methods=['GET', 'POST'])
@cross_origin()
def face_detection_route():
    global face_cap, camera_active
    if request.method == 'POST':
        with camera_lock:
            if camera_active:
                flash("Camera is already in use for another operation.", "danger")
                return render_template('face_detection.html')

            camera_active = True
            frame = capture_frame(face_cap)
            if frame is not None:
                processed_frame = detect_faces(frame)
                output_path = os.path.join('static', 'face_detection.jpg')
                cv2.imwrite(output_path, processed_frame)

                # Release the camera after processing
                if face_cap is not None:
                    face_cap.release()
                    face_cap = None

                camera_active = False
                return send_file(output_path, mimetype='image/jpeg')

            else:
                flash("Failed to capture image.", "danger")
                camera_active = False

    return render_template('face_detection.html')


@app.route('/face-training', methods=['GET', 'POST'])
@cross_origin()
def face_training_route():
    return render_template('face_training.html')

@app.route('/object-detection', methods=['GET', 'POST'])
@cross_origin()
def object_detection():
    if request.method == 'POST':
        source = request.form.get('source')
        if source == 'upload' and 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash("No file selected.", "danger")
                return render_template('object_detection.html')
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            frame = cv2.imread(file_path)
        elif source == 'esp32':
            esp32_url = request.form.get('esp32_url')
            if not esp32_url:
                flash("No ESP32 URL provided.", "danger")
                return render_template('object_detection.html')
            return render_template('object_detection.html', live="/video_feed_esp32?url=" + esp32_url)
        elif source == 'camera':
            return render_template('object_detection.html', live="/video_feed_object_detection")
        else:
            frame = capture_frame()
        if frame is not None:
            processed_frame = detect_objects(frame)
            output_path = os.path.join('static', 'object_detection.jpg')
            cv2.imwrite(output_path, processed_frame)
            return redirect(url_for('object-detection', output=output_path))
        else:
            flash("Failed to capture image.", "danger")
    return render_template('object_detection.html')

@app.route('/upload_folder', methods=['POST'])
@cross_origin()
def upload_folder():
    trainee_name = request.form.get('traineeNameUpload', '').strip()
    if not trainee_name:
        flash("Please enter your name for training.", "danger")
        return redirect(url_for('face_training_route'))
    files = request.files.getlist('face_folder')
    if not files:
        flash("No folder or files were uploaded.", "danger")
        return redirect(url_for('face_training_route'))
    trainee_folder = os.path.join('data', secure_filename(trainee_name))
    os.makedirs(trainee_folder, exist_ok=True)
    count = 0
    for file in files:
        if file and file.filename:
            try:
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"⚠️ Could not decode image: {file.filename}")
                    continue
                face_crop = detect_and_crop_face(img)
                if face_crop is None:
                    print(f"⚠️ No face detected in image: {file.filename}")
                    continue
                count += 1
                filename = os.path.join(trainee_folder, f"{count}.png")
                cv2.imwrite(filename, face_crop)
            except Exception as e:
                print(f"❌ Error processing file {file.filename}: {e}")
    if count > 0:
        threading.Thread(target=train_faces).start()
        flash(f"Uploaded and processed {count} image(s) for '{trainee_name}'. Training started.", "success")
    else:
        flash("No valid face images were processed from the uploaded folder.", "danger")
    return redirect(url_for('home'))

@app.route('/mobile_cam')
@cross_origin()
def mobile_cam():
    return render_template('mobile_camera.html')

@app.route('/login/google/authorized')
@cross_origin()
def google_authorized():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google.", "danger")
        return redirect(url_for("signin"))
    user_info = resp.json()
    email = user_info.get("email", "Unknown")
    name = user_info.get("name", "")
    
    # If the user exists in the database, log them in
    if email in users:
        session['user'] = email
        flash("Logged in with Google successfully!", "success")
        return redirect(url_for('home'))
    else:
        # Otherwise, store their info in session and redirect to signup with name prefilled
        session['google_email'] = email
        session['google_name'] = name
        flash("No account found. Please sign up to complete registration.", "warning")
        return redirect(url_for('signup'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
