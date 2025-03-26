import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
const String baseUrl = 'http://127.0.0.1:5000';
void main() {
  runApp(SmartSightApp());
}

class SmartSightApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartSight',
      theme: lightTheme,
      darkTheme: darkTheme,
   home: MyHomePage(),// Start with the Splash Screen
    );
  }
}

// =========================
// Splash Screen
// =========================
class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _navigateToHome();
  }

  _navigateToHome() async {
    await Future.delayed(Duration(seconds: 3), () {});
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => HomeScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset('assets/icon1.png', height: 150),
            SizedBox(height: 20),
            Text(
              'SmartSight',
              style: TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// =========================
// Home Screen
// =========================
class HomeScreen extends StatelessWidget {
  final GoogleSignIn _googleSignIn = GoogleSignIn();

  Future<void> _handleGoogleSignIn(BuildContext context) async {
    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) {
        print('Google Sign-In cancelled');
        return;
      }

      print('Signed in as: ${googleUser.displayName}');

      // Navigate to DashboardScreen after successful login
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => MyHomePage()),
      );
    } catch (error) {
      print('Error signing in: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('SmartSight'),
      ),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/2.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.5),
          ),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'See the Unseen, Feel the Vision',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () => _handleGoogleSignIn(context),
                  child: Text('Sign in with Google'),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => LoginPage()),
                    );
                  },
                  child: Text('Sign in with Email'),
                ),
                SizedBox(height: 10),
                TextButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SignUpPage()),
                    );
                  },
                  child: Text(
                    'Create an Account',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// =========================
// Login Page
// =========================
class LoginPage extends StatefulWidget {
  const LoginPage({Key? key}) : super(key: key);

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _login() {
    final email = _emailController.text;
    final password = _passwordController.text;

    if (email.isNotEmpty && password.isNotEmpty) {
      // Navigate to the main app after successful login
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const MyHomePage(),
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please fill in all fields')),
      );
    }
  }

  void _navigateToSignUp() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const SignUpPage(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Login'),
      ),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/2.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.5),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                TextField(
                  controller: _emailController,
                  decoration: const InputDecoration(
                    labelText: 'Email',
                    border: OutlineInputBorder(),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),
                TextField(
                  controller: _passwordController,
                  decoration: const InputDecoration(
                    labelText: 'Password',
                    border: OutlineInputBorder(),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                  obscureText: true,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _login,
                  child: const Text('Login'),
                ),
                const SizedBox(height: 10),
                TextButton(
                  onPressed: _navigateToSignUp,
                  child: const Text(
                    'Don\'t have an account? Sign Up',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// =========================
// Sign Up Page
// =========================
class SignUpPage extends StatefulWidget {
  const SignUpPage({Key? key}) : super(key: key);

  @override
  State<SignUpPage> createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _signUp() {
    final email = _emailController.text;
    final password = _passwordController.text;

    if (email.isNotEmpty && password.isNotEmpty) {
      // Navigate to the main app after successful sign-up
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const MyHomePage(),
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please fill in all fields')),
      );
    }
  }

  void _navigateToLogin() {
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sign Up'),
      ),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/2.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.5),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                TextField(
                  controller: _emailController,
                  decoration: const InputDecoration(
                    labelText: 'Email',
                    border: OutlineInputBorder(),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),
                TextField(
                  controller: _passwordController,
                  decoration: const InputDecoration(
                    labelText: 'Password',
                    border: OutlineInputBorder(),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                  obscureText: true,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _signUp,
                  child: const Text('Sign Up'),
                ),
                const SizedBox(height: 10),
                TextButton(
                  onPressed: _navigateToLogin,
                  child: const Text(
                    'Already have an account? Login',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// =========================
// Main App (MyHomePage)
// =========================
class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key}) : super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  
  // ============ OBJECT DETECTION ============
  Future<void> detectObjects() async {
    final response = await http.post(Uri.parse('$baseUrl/detect_objects'));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      _showResult('Object Detection', data['status']);
    } else {
      _showResult('Error', 'Failed to detect objects');
    }
  }

  // ============ FACE RECOGNITION ============
  Future<void> recognizeFace() async {
    final response = await http.post(Uri.parse('$baseUrl/recognize_face'));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      _showResult('Face Recognition', "Name: ${data['name']}, Confidence: ${data['confidence']}");
    } else {
      _showResult('Error', 'Failed to recognize face');
    }
  }

  // ============ FACE TRAINING ============
  Future<void> trainFace() async {
    final response = await http.post(Uri.parse('$baseUrl/train_face'));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      _showResult('Face Training', data['status']);
    } else {
      _showResult('Error', 'Failed to train face');
    }
  }

  // ============ TEXT TO SPEECH ============
  Future<void> readText() async {
    final response = await http.post(
      Uri.parse('$baseUrl/read_text'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': 'Hello, this is a sample text.'}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      _showResult('Text-to-Speech', data['status']);
    } else {
      _showResult('Error', 'Failed to read text');
    }
  }

  // ============ START CAMERA ============
  Future<void> startCamera() async {
    final response = await http.get(Uri.parse('$baseUrl/start_camera'));
    if (response.statusCode == 200) {
      _showResult('Camera', 'Camera started');
    } else {
      _showResult('Error', 'Failed to start camera');
    }
  }

  // ============ STOP CAMERA ============
  Future<void> stopCamera() async {
    final response = await http.get(Uri.parse('$baseUrl/stop_camera'));
    if (response.statusCode == 200) {
      _showResult('Camera', 'Camera stopped');
    } else {
      _showResult('Error', 'Failed to stop camera');
    }
  }

  // ============ SHOW RESULT DIALOG ============
  void _showResult(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SmartSight')),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: detectObjects,
              child: const Text('Object Detection'),
            ),
            ElevatedButton(
              onPressed: recognizeFace,
              child: const Text('Face Recognition'),
            ),
            ElevatedButton(
              onPressed: trainFace,
              child: const Text('Face Training'),
            ),
            ElevatedButton(
              onPressed: readText,
              child: const Text('Text-to-Speech'),
            ),
            ElevatedButton(
              onPressed: startCamera,
              child: const Text('Start Camera'),
            ),
            ElevatedButton(
              onPressed: stopCamera,
              child: const Text('Stop Camera'),
            ),
          ],
        ),
      ),
    );
  }
}


  void _increaseVolume() {
    setState(() {
      if (_volumeLevel < 100) _volumeLevel += 10;
    });
  }

  void _decreaseVolume() {
    setState(() {
      if (_volumeLevel > 0) _volumeLevel -= 10;
    });
  }

  void _showLogoutConfirmation(BuildContext context) async {
    final GoogleSignIn _googleSignIn = GoogleSignIn();

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Confirm Logout'),
          content: const Text('Are you sure you want to log out?'),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(); // Close the dialog
              },
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () async {
                Navigator.of(context).pop(); // Close the dialog

                // Sign out from Google
                await _googleSignIn.signOut();

                // Clear the navigation stack and go to LoginPage
                Navigator.pushAndRemoveUntil(
                  context,
                  MaterialPageRoute(builder: (context) => const LoginPage()),
                  (Route<dynamic> route) => false, // Remove all routes
                );
              },
              child: const Text('Logout'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: _currentTheme,
      home: Scaffold(
        appBar: AppBar(
          title: Row(
            children: [
              Image.asset(
                'assets/icon1.png',
                height: 40,
              ),
              const SizedBox(width: 10),
              const Text(
                'SmartSight',
                style: TextStyle(
                  fontFamily: 'Roboto',
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ],
          ),
          actions: [
            IconButton(
              icon: const Icon(Icons.brightness_6),
              onPressed: _toggleTheme,
            ),
            PopupMenuButton<String>(
              onSelected: (String value) {
                if (value == 'Settings') {
                  print('Settings selected');
                } else if (value == 'Account') {
                  print('Account selected');
                } else if (value == 'Help') {
                  print('Help selected');
                }
              },
              itemBuilder: (BuildContext context) {
                return [
                  const PopupMenuItem<String>(
                    value: 'Settings',
                    child: Text('Settings'),
                  ),
                  const PopupMenuItem<String>(
                    value: 'Account',
                    child: Text('Account'),
                  ),
                  const PopupMenuItem<String>(
                    value: 'Help',
                    child: Text('Help'),
                  ),
                ];
              },
              icon: const Icon(Icons.menu),
            ),
            IconButton(
              icon: const Icon(Icons.logout),
              onPressed: () => _showLogoutConfirmation(context),
            ),
          ],
        ),
        body: Container(
          decoration: const BoxDecoration(
            image: DecorationImage(
              image: AssetImage('assets/2.jpg'),
              fit: BoxFit.cover,
            ),
          ),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.5),
            ),
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'See the Unseen, Feel the Vision',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 20),
                  _buildBatteryIndicator(),
                  const SizedBox(height: 20),
                  _buildConnectionStatus(),
                  const SizedBox(height: 20),
                  _buildVolumeControls(),
                  const SizedBox(height: 40),
                  _buildButton(
                    icon: Icons.visibility,
                    label: "Object Detection & Avoidance",
                    onPressed: () => print('Object detection activated'),
                  ),
                  const SizedBox(height: 15),
                  _buildButton(
                    icon: Icons.text_fields,
                    label: "Text Reading",
                    onPressed: () => print('Text reading activated'),
                  ),
                  const SizedBox(height: 15),
                  _buildButton(
                    icon: Icons.face,
                    label: "Face Recognition",
                    onPressed: () => print('Face recognition activated'),
                  ),
                  const SizedBox(height: 15),
                  _buildButton(
                    icon: Icons.face_retouching_natural,
                    label: "Face Training",
                    onPressed: () => print('Face training activated'),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBatteryIndicator() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.battery_full,
          color: _batteryLevel > 50
              ? Colors.green
              : _batteryLevel > 20
                  ? Colors.yellow
                  : Colors.red,
          size: 30,
        ),
        const SizedBox(width: 10),
        Text(
          'Battery: $_batteryLevel%',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
          ),
        ),
      ],
    );
  }

  Widget _buildConnectionStatus() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          _isDeviceConnected ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
          color: _isDeviceConnected ? Colors.green : Colors.red,
          size: 30,
        ),
        const SizedBox(width: 10),
        Text(
          _isDeviceConnected ? 'Connected' : 'Disconnected',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
          ),
        ),
        const SizedBox(width: 10),
        ElevatedButton(
          onPressed: _toggleConnection,
          child: Text(_isDeviceConnected ? 'Disconnect' : 'Connect'),
        ),
      ],
    );
  }

  Widget _buildVolumeControls() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        IconButton(
          icon: const Icon(Icons.volume_down, size: 30),
          onPressed: _decreaseVolume,
        ),
        Text(
          'Volume: $_volumeLevel%',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
          ),
        ),
        IconButton(
          icon: const Icon(Icons.volume_up, size: 30),
          onPressed: _increaseVolume,
        ),
      ],
    );
  }

  Widget _buildButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 60,
      child: ElevatedButton.icon(
        icon: Icon(icon, size: 28),
        label: Text(label),
        onPressed: onPressed,
      ),
    );
  }
}

// Light Theme
final lightTheme = ThemeData(
  brightness: Brightness.light,
  primaryColor: Colors.blue,
);

// Dark Theme
final darkTheme = ThemeData(
  brightness: Brightness.dark,
  primaryColor: Colors.blueGrey,
);
