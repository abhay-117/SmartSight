
  <h1>👓 SmartSight</h1>

  <p><strong>SmartSight</strong> is an intelligent assistive system designed to enhance the independence and awareness of visually impaired individuals. By integrating embedded systems, computer vision, and mobile technology, it enables real-time object detection, text recognition, and face identification — with instant audio feedback via a companion Flutter mobile app.</p>

  <section>
    <h2>✨ Features</h2>
    <ul>
      <li><strong>🔍 Real-Time Object Detection</strong> – Detect obstacles and objects using YOLOv8 and ESP32-CAM</li>
      <li><strong>🧠 Face Recognition</strong> – Identify familiar individuals with personalized face recognition</li>
      <li><strong>📖 Text-to-Speech (OCR)</strong> – Convert printed or handwritten text into spoken words</li>
      <li><strong>📱 Flutter Mobile App</strong> – Simple interface for control, interaction, and feedback</li>
      <li><strong>📡 Wireless ESP32-CAM Streaming</strong> – Low-power, compact vision module with real-time streaming</li>
      <li><strong>🧩 Modular System Design</strong> – Each core function runs independently for flexibility and scalability</li>
    </ul>
  </section>

  <section>
    <h2>📁 Repository Structure</h2>
    <pre><code>
smartsight/
├── esp32/                 # ESP32-CAM code for object detection
├── flutter_app/           # Flutter mobile app source
├── ocr_module/            # Optical Character Recognition (OCR) + TTS
├── face_recognition/      # Face recognition & training scripts
├── models/                # Pretrained YOLOv8/other models
├── utils/                 # Helper scripts, configs
└── README.md              # Documentation
    </code></pre>
  </section>

  <section>
    <h2>🚀 Getting Started</h2>
    <ol>
      <li><strong>Clone the Repository</strong>
        <pre><code>git clone https://github.com/yourusername/smartsight.git
cd smartsight</code></pre>
      </li>

      <li><strong>Python Environment Setup</strong>
        <pre><code># Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ocr_module/requirements.txt
pip install -r face_recognition/requirements.txt
pip install -r utils/requirements.txt</code></pre>
      </li>

      <li><strong>ESP32-CAM Setup</strong>
        <ul>
          <li>Open <code>/esp32/</code> using Arduino IDE or PlatformIO</li>
          <li>Set your Wi-Fi SSID and password in the code</li>
          <li>Flash to ESP32-CAM board</li>
          <li>Access the live camera stream via the board's IP address</li>
        </ul>
      </li>

      <li><strong>Flutter App Setup</strong>
        <pre><code>cd flutter_app
flutter pub get
flutter run</code></pre>
        <p>✅ Make sure Flutter and Android Studio (or a connected device/emulator) are properly set up.</p>
      </li>
    </ol>
  </section>

  <section>
    <h2>🛠️ Usage</h2>
    <ul>
      <li><strong>🖼️ Object Detection</strong> – ESP32 streams to mobile app → YOLOv8 processes images → Audio feedback to user</li>
      <li><strong>👤 Face Recognition</strong> – <code>face_recognition/recognize.py</code> runs in the background to identify people</li>
      <li><strong>📝 Text Reader (OCR + TTS)</strong> – <code>ocr_module/main.py</code> captures and reads text aloud</li>
      <li><strong>📲 Mobile Interface</strong> – Users trigger modules, receive alerts, and switch modes using the app</li>
    </ul>
  </section>

</body>
</html>
