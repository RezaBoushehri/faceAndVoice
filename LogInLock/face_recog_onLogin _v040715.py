import cv2
import numpy as np
import pymongo
import base64
import threading
from PIL import Image
import io
import face_recognition
from datetime import datetime, timedelta
import time
import dlib  # Make sure to install dlib
from scipy.spatial import distance  # For EAR calculation
import pyttsx3  # For text-to-speech; install with: pip install pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# from voice_recognation.recognize_speakerTorch import recognize_speaker
import os

# Define your desired frame dimensions
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1080

# Initialize HOG descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this file is present

# MongoDB connection
client = pymongo.MongoClient('mongodb://root:admin@localhost:27017/')
db = client['face_recognition']
collection = db['users']

# Email configuration - Replace with your actual SMTP details
SMTP_SERVER = 'smtp.gmail.com'  # Example: Gmail
SMTP_PORT = 465
SENDER_EMAIL = 'boshehry.reza@gmail.com'
SENDER_PASSWORD = 'vyylcmcjovuyrfcj'  # Use app password for Gmail
RECIPIENT_EMAIL = 'reza.boshehry@gmail.com'
ALERT_SUBJECT = 'WARNING Unauthorized Access Detected'

# Lock timeout in minutes
LOCK_TIMEOUT_MINUTES = 1

# Face recognition tolerance (lower = stricter matching)
TOLERANCE = 0.5  # Default is 0.6; lower for higher security

# Maximum failed attempts before permanent lock or logout
MAX_FAILED_ATTEMPTS = 3

# Load known faces and names from MongoDB
def load_known_faces():
    known_faces = []
    known_names = []
    for entry in collection.find():
        name = entry['name']
        images = entry['images']
        for key in images.keys():
            image_data = base64.b64decode(images[key].split(",")[1])
            img = Image.open(io.BytesIO(image_data))
            img = np.array(img)
            face_encoding = face_recognition.face_encodings(img)
            if face_encoding:
                known_faces.append(face_encoding[0])
                known_names.append(name)
    return known_faces, known_names

known_faces, known_names = load_known_faces()

class VideoCamera:
    # def __init__(self, source="rtsp://admin:farahoosh@3207@172.16.28.6"):
    def __init__(self, source=0):
        self.vid = cv2.VideoCapture(source)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

    def update_frame(self):
        while self.running:
            success, frame = self.vid.read()
            if success:
                # Resize the frame
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                self.frame = frame
            time.sleep(0.03)  # Adjust sleep to reduce CPU usage

    def get_frame(self):
        return self.frame

    def __del__(self):
        self.running = False
        self.vid.release()

camera = VideoCamera()

class FaceRecognitionApp:
    def __init__(self):
        self.detecting = False
        self.face_detected = False
        self.running = True
        self.locked = False  # Flag for locked state
        self.last_valid_detection = datetime.now()  # Track last valid face detection time
        self.failed_attempts = 0  # Counter for failed unlock attempts

    def run(self):
        while self.running:
            frame = camera.get_frame()
            if frame is not None:
                # Check for lock timeout
                if not self.locked and (datetime.now() - self.last_valid_detection) > timedelta(minutes=LOCK_TIMEOUT_MINUTES):
                    self.lock_system()
                    print("System locked due to inactivity.")

                if not self.face_detected:
                    # Reduce frame resolution to 640x480 for lower resource usage
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect only if human is detected and face not logged recently
                    if self.detect_human(rgb_frame):
                        self.capture_and_detect(rgb_frame)

            time.sleep(0.033)  # Equivalent to 30 FPS update

    def detect_human(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8))
        return len(humans) > 0

    def capture_and_detect(self, frame):
        # Always capture frames for liveness detection
        frames = []
        for _ in range(3):  # Capture 3 frames for analysis
            captured_frame = camera.get_frame()
            if captured_frame is not None:
                small_frame = cv2.resize(captured_frame, (640, 480))  # Lower resolution for speed
                frames.append(small_frame.copy())
            time.sleep(0.1)

        alive = self.analyze_frames(frames)
        if not alive:
            print("Liveness detection failed: Possible spoofing attempt.")
            if self.locked:
                self.failed_attempts += 1
                if self.failed_attempts >= MAX_FAILED_ATTEMPTS:
                    self.permanent_lock()
            return

        if self.locked:
            # If locked, only allow unlocking with valid face + liveness
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            detected_faces = set()

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
                name = "Unknown"

                if True in matches:
                    face_distances = face_recognition.face_distance(known_faces, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < TOLERANCE:
                        name = known_names[best_match_index]
                        detected_faces.add(name)
                    else:
                        detected_faces.add('Unknown')
                else:
                    detected_faces.add('Unknown')

                # Draw rectangles and labels (optional)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Log detected faces only if valid
            for name in detected_faces:
                if name == "BB":
                    self.log_alive_status(alive, name, frames)
                    self.failed_attempts = 0  # Reset on success
                    return  # Unlock and grant access

            # Failed attempt
            self.failed_attempts += 1
            print(f"Failed unlock attempt: {self.failed_attempts}/{MAX_FAILED_ATTEMPTS}")
            if self.failed_attempts >= MAX_FAILED_ATTEMPTS:
                self.permanent_lock()
            self.play_tts("Access denied. Too many failed attempts.")
            return

        # Normal detection flow (unlocked state)
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Prepare to log detected faces
        detected_faces = set()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            if True in matches:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < TOLERANCE:
                    name = known_names[best_match_index]
                    detected_faces.add(name)
                else:
                    detected_faces.add('Unknown')
            else:
                detected_faces.add('Unknown')

            # Draw rectangles and labels (optional)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Log detected faces
        for name in detected_faces:
            self.log_alive_status(alive, name, frames)

    def analyze_frames(self, frames):
        """Analyze the frames to check for blinking, mouth movement, or head rotation."""
        if len(frames) < 2:
            return False  # Not enough frames to analyze

        alive = False
        blink_count = 0
        motion_detected = False
        previous_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        blink_frames = 0

        for i in range(1, len(frames)):
            current_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Motion detection with stricter threshold for security
            frame_diff = cv2.absdiff(previous_frame, current_frame)
            _, thresh = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)  # Slightly stricter

            non_zero_count = np.sum(thresh)
            if non_zero_count > 8000:  # Higher threshold for more reliable motion
                motion_detected = True

            previous_frame = current_frame

            # Detect faces in the current frame
            faces = detector(current_frame)

            if faces:
                shape = predictor(current_frame, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                # Get the eye landmarks
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]

                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                # Stricter EAR threshold for blink detection
                if left_ear < 0.25 or right_ear < 0.25:  # Lower threshold for better anti-spoofing
                    blink_frames += 1

                if blink_frames >= 1:  # At least 1 frame blink
                    blink_count += 1
                    blink_frames = 0

        # Require both blink and motion for higher security
        if blink_count > 0 and motion_detected:
            alive = True

        return alive

    def log_face(self, name):
        log_data = {"name": name, "time": datetime.now()}
        print(f"Logged {name} at {log_data['time']}")

    def log_alive_status(self, alive, name, frames):
        log_data = {
            "alive": alive,
            "timestamp": datetime.now()
        }
        status = "alive" if alive else "not alive"
        print(f"Logged: {name} is {status} at {log_data['timestamp']}")

        if name == "BB":
            self.last_valid_detection = datetime.now()  # Update last detection time
            self.locked = False  # Unlock if was locked
            # Save picture in magels folder with timestamp
            if frames:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp_str}.jpg"
                os.makedirs("magels", exist_ok=True)
                cv2.imwrite(f"magels/{filename}", frames[-1])
            self.play_tts("Welcome Reza Boushehri, The best Developer in the world. access granted.")
            # App continues running
        else:
            # For unknown faces, deny access: send alert and lock (if not already locked)
            if not self.locked:
                self.send_email_alert(name)
                self.lock_system()
            else:
                # If already locked, just log the attempt (no additional alert to avoid spam)
                print(f"Access denied for {name} (system already locked).")

    def lock_system(self):
        """Lock the current user session."""
        self.locked = True
        import platform
        system = platform.system()
        if system == "Windows":
            os.system("rundll32.exe user32.dll,LockWorkStation")  # Locks the workstation
        elif system == "Linux":
            os.system("xdg-screensaver lock")  # Example for Linux; adjust as needed
        else:
            print("Lock not supported on this OS.")
        print("System locked.")

    def permanent_lock(self):
        """Handle permanent lock after max failed attempts."""
        self.play_tts("System permanently locked. Contact administrator.")
        self.send_email_alert("Multiple failed attempts - Permanent lock activated")
        # Optional: Force logout or shutdown for extreme security
        import platform
        system = platform.system()
        if system == "Windows":
            os.system("shutdown /l")  # Force logout
        elif system == "Linux":
            os.system("gnome-session-quit --power-off")  # Force shutdown; adjust as needed
        self.running = False  # Stop the app

    def play_tts(self, text):
        """Play text-to-speech."""
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def send_email_alert(self, name):
        """Send email alert via SMTP."""
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = ALERT_SUBJECT

            body = f"Unauthorized person detected: {name}. Timestamp: {datetime.now()}. Action: System locked."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
            server.quit()
            print(f"Email alert sent for {name}")
        except Exception as e:
            print(f"Email error: {e}")

    def eye_aspect_ratio(self, eye):
        """Calculate the Eye Aspect Ratio (EAR) for an eye given its landmarks."""
        A = distance.euclidean(eye[1], eye[5])  # 6th point - 2nd point
        B = distance.euclidean(eye[2], eye[4])  # 5th point - 3rd point
        C = distance.euclidean(eye[0], eye[3])  # 4th point - 1st point
        ear = (A + B) / (2.0 * C)
        return ear

if __name__ == "__main__":
    app = FaceRecognitionApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("App stopped by user.")
    finally:
        camera.__del__()