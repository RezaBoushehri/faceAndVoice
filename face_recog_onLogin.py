import cv2
import numpy as np
import pymongo
import base64
import threading
from PIL import Image
import io
import face_recognition
from datetime import datetime
import time
import dlib  # Make sure to install dlib with CUDA support for GPU acceleration
from scipy.spatial import distance  # For EAR calculation
import pyttsx3  # For text-to-speech; install with: pip install pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# from voice_recognation.recognize_speakerTorch import recognize_speaker
import os
import msvcrt  # For non-blocking key detection (Windows)
import tkinter as tk
from tkinter import ttk
import subprocess
import sys

# Define your desired frame dimensions
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1080

# Initialize HOG descriptor for human detection (CPU-based, fast enough for initial detection)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load dlib's face detector and the facial landmark predictor
# Note: For GPU acceleration, compile dlib with CUDA support (see: http://dlib.net/compile.html)
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

# Cache for known faces (loaded lazily)
known_faces = []
known_names = []
precompute_done = False

# Backdoor hidden password sequence
HIDDEN_PASSWORD = "Rez@2020"
input_buffer = ""  # Buffer for typed characters

# Global lock screen window
lock_window = None

def disable_usb_ports():
    """Disable USB ports (requires admin privileges). Run as admin."""
    try:
        # PowerShell command to disable USB storage devices
        ps_command = """
        Get-PnpDevice -FriendlyName "*USB*" | Where-Object {$_.Status -eq 'OK'} | Disable-PnpDevice -Confirm:$false
        """
        subprocess.run(['powershell', '-Command', ps_command], check=True, capture_output=True)
        print("USB ports disabled.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to disable USB ports (run as admin): {e}")

def enable_usb_ports():
    """Enable USB ports (requires admin privileges)."""
    try:
        # PowerShell command to enable USB storage devices
        ps_command = """
        Get-PnpDevice -FriendlyName "*USB*" | Where-Object {$_.Status -eq 'Error'} | Enable-PnpDevice -Confirm:$false
        """
        subprocess.run(['powershell', '-Command', ps_command], check=True, capture_output=True)
        print("USB ports enabled.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to enable USB ports (run as admin): {e}")

def create_lock_screen():
    """Create a full-screen blank black window to lock the screen."""
    global lock_window
    lock_window = tk.Tk()
    lock_window.title("Secure Login")
    lock_window.attributes('-fullscreen', True)
    lock_window.configure(bg='black')
    lock_window.attributes('-topmost', True)
    # Hide taskbar (Windows-specific, approximate)
    lock_window.overrideredirect(True)  # Remove window borders
    # Bind escape or other keys if needed, but keep keyboard input for hidden pass
    lock_window.bind('<Escape>', lambda e: None)  # Do nothing for escape
    lock_window.mainloop()  # This blocks, so run in thread if needed

def precompute_encodings():
    """Precompute face encodings for all images per user and store them in MongoDB for faster loading."""
    global precompute_done
    print("Starting precomputation of encodings for all images...")
    updated_count = 0
    needs_update = False
    for entry in collection.find():
        if 'encodings' not in entry:
            needs_update = True
            break
    if not needs_update:
        print("Encodings already precomputed for all images. Skipping.")
        precompute_done = True
        return

    for entry in collection.find():
        if 'encodings' not in entry:
            encodings = []
            name = entry['name']
            images = entry['images']
            for key in images.keys():
                try:
                    # Extract base64 payload (skip data URL prefix if present)
                    img_str = images[key]
                    if img_str.startswith('data:image'):
                        image_data = base64.b64decode(img_str.split(",")[1])
                    else:
                        image_data = base64.b64decode(img_str)
                    
                    img = Image.open(io.BytesIO(image_data)).convert('RGB')
                    img = np.array(img)
                    face_encoding = face_recognition.face_encodings(img, model='hog')  # Use HOG for faster precompute
                    if face_encoding:
                        encodings.append(face_encoding[0].tolist())  # Store as list of floats
                        print(f"Precomputed encoding for {key} of {name}")
                except Exception as e:
                    print(f"Error processing image {key} for {name}: {e}")
            if encodings:
                collection.update_one({'_id': entry['_id']}, {'$set': {'encodings': encodings}})
                updated_count += 1
                print(f"Precomputed {len(encodings)} encodings for all images of {name}")
    print(f"Precomputation complete. Updated {updated_count} users with all images processed.")
    precompute_done = True

def load_known_faces(use_precomputed=True):
    """Load known faces and names from MongoDB, preferring precomputed if available."""
    global known_faces, known_names
    known_faces = []
    known_names = []
    for entry in collection.find():
        name = entry['name']
        if 'encodings' in entry and use_precomputed:
            for enc_list in entry['encodings']:
                known_faces.append(np.array(enc_list))
                known_names.append(name)
        else:
            # Fallback: compute on the fly for all images (no break for consistency)
            print(f"Computing encodings on-the-fly for all images of {name}...")
            images = entry['images']
            for key in images.keys():
                try:
                    # Extract base64 payload (skip data URL prefix if present)
                    img_str = images[key]
                    if img_str.startswith('data:image'):
                        image_data = base64.b64decode(img_str.split(",")[1])
                    else:
                        image_data = base64.b64decode(img_str)
                    
                    img = Image.open(io.BytesIO(image_data)).convert('RGB')
                    img = np.array(img)
                    face_encoding = face_recognition.face_encodings(img, model='hog')
                    if face_encoding:
                        known_faces.append(face_encoding[0])
                        known_names.append(name)
                except Exception as e:
                    print(f"Error loading image {key} for {name}: {e}")
    print(f"Loaded {len(known_faces)} known faces from all images.")

# Start and wait for precomputation to complete on startup (blocks until done, but fast if already precomputed)
precompute_thread = threading.Thread(target=precompute_encodings, daemon=True)
precompute_thread.start()
precompute_thread.join()  # Wait for completion to ensure encodings are ready before proceeding

# Now load with precomputed (since precompute_done is True)
load_known_faces(use_precomputed=True)

class VideoCamera:
    def __init__(self, source=0):
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            print(f"Error: Could not open video source {source}")
            self.vid = cv2.VideoCapture(0)  # Fallback to webcam
            print("Fallback to default webcam (source=0)")
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
                # print(f"Frame captured successfully: shape {frame.shape}")  # Debug print to confirm capture
            else:
                print("Failed to capture frame")  # Debug print
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
        self.last_detection_time = 0  # To avoid reloading faces too frequently
        self.access_granted = False

    def check_hidden_backdoor(self):
        """Check for hidden backdoor password in input buffer."""
        global input_buffer
        while msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8')
            input_buffer += key
            # Optional: Limit buffer size to prevent memory issues
            if len(input_buffer) > 100:
                input_buffer = input_buffer[-50:]  # Keep last 50 chars
        if HIDDEN_PASSWORD in input_buffer:
            print("Hidden backdoor activated.")
            self.play_tts("Welcome Reza Boushehri, access granted ")
            input_buffer = ""  # Clear buffer after success
            self.grant_access()
            return True
        return False

    def grant_access(self):
        """Grant access: Enable USB, close lock screen, exit app."""
        self.access_granted = True
        enable_usb_ports()
        if lock_window:
            lock_window.destroy()
        self.close_app()

    def run(self):
        global lock_window
        # Disable USB ports on start
        disable_usb_ports()
        # Create lock screen in a separate thread (since mainloop blocks)
        lock_thread = threading.Thread(target=create_lock_screen, daemon=True)
        lock_thread.start()
        print("Starting face recognition loop...")  # Debug print
        print("Hidden backdoor active: Type a sequence containing 'Rez@2020' anywhere to grant access.")  # Instruction for backdoor
        while self.running and not self.access_granted:
            # Always check for hidden backdoor (works during face processing too)
            if self.check_hidden_backdoor():
                return  # Exit if backdoor triggered

            frame = camera.get_frame()
            if frame is not None:
                # Throttle processing to every 10th frame for speed
                if int(time.time() * 10) % 10 == 0:  # Process ~10 FPS
                    # Downscale frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Detect only if human is detected and face not logged recently
                    if self.detect_human(rgb_frame):
                        print("Human detected, starting face recognition...")  # Debug print
                        self.capture_and_detect(small_frame)
                        if self.access_granted:
                            break

            time.sleep(0.1)  # Slightly higher sleep for less CPU

    def detect_human(self, frame):
        # Further downscale for human detection
        tiny_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray_frame = cv2.cvtColor(tiny_frame, cv2.COLOR_BGR2GRAY)
        humans, _ = hog.detectMultiScale(gray_frame, winStride=(4, 4), scale=1.05)  # Adjusted for smaller frame
        print(f"Human detection: {len(humans)} humans found")  # Debug print
        return len(humans) > 0

    def capture_and_detect(self, frame):
        global known_faces, known_names, precompute_done
        # Since precompute is done on startup, no need to reload unless empty (edge case)
        if not known_faces:
            print(f"load_known_faces in capture_and_detect") 
            load_known_faces(use_precomputed=True)

        # Use HOG model for faster CPU processing (instead of CNN)
        face_locations = face_recognition.face_locations(frame, model="hog")
        face_encodings = face_recognition.face_encodings(frame, face_locations, model="hog")

        # Prepare to log detected faces
        detected_faces = set()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)  # Slightly looser tolerance for speed/accuracy trade-off
            name = "Unknown"

            if True in matches:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= 0.6:
                    name = known_names[best_match_index]
                    detected_faces.add(name)
                else:
                    detected_faces.add('Unknown')
            else:
                detected_faces.add('Unknown')

            # Draw rectangles and labels (optional, for logging or debugging; can be removed if not needed)
            # Scale back for drawing if needed, but since small frame, optional
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)  # Smaller font for small frame

        # ** Capture frames for movement analysis if a human is detected **
        frames = []
        for _ in range(30):  # Reduced to 30 frames for speed
            captured_frame = camera.get_frame()
            if captured_frame is not None:
                # Downscale captured frames
                small_captured = cv2.resize(captured_frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2))
                frames.append(small_captured.copy())  # Use copy to preserve the frame
                time.sleep(0.033)  # Slightly faster capture

        # Analyze frames once (shared for all detected faces)
        alive = self.analyze_frames(frames)

        # Log detected faces (common log for all)
        for name in detected_faces:
            self.log_face(name)
            log_data = {
                "alive": alive,
                "timestamp": datetime.now()
            }
            status = "alive" if alive else "not alive"
            print(f"Logged: {name} is {status} at {log_data['timestamp']}")

        # if alive:
        if True:
            if "BB" in detected_faces:
                # If BB is among detected faces, grant access
                self.play_tts("Welcome Reza Boushehri, access granted.")
                self.grant_access()
            else:
                # Unauthorized: alert and logout
                # Save picture in magels folder with timestamp (use last frame, full size)
                if frames:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    last_full = camera.get_frame()  # Get full size for save
                    for name in detected_faces:
                        if name != "Unknown":  # Only save for known unauthorized
                            filename = f"{name}_{timestamp_str}.jpg"
                            os.makedirs("magels", exist_ok=True)
                            cv2.imwrite(f"magels/{filename}", last_full)
                    self.send_email_alert(list(detected_faces))  # Send for all unauthorized
                self.logout_system()

    def log_face(self, name):
        log_data = {"name": name, "time": datetime.now()}
        print(f"Logged {name} at {log_data['time']}")

    def close_app(self):
        self.running = False

    def play_tts(self, text):
        """Play text-to-speech."""
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def send_email_alert(self, names):
        """Send email alert via SMTP for multiple names."""
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = ALERT_SUBJECT

            names_str = ", ".join(names)
            body = f"Unauthorized persons detected: {names_str}. Timestamp: {datetime.now()}. Action: Logging out."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
            server.quit()
            print(f"Email alert sent for {names}")
        except Exception as e:
            print(f"Email error: {e}")

    def logout_system(self):
        """Log out the current user. This is OS-specific; example for Windows."""
        import platform
        system = platform.system()
        if system == "Windows":
            os.system("shutdown /l")  # Logs out current user
        elif system == "Linux":
            os.system("gnome-session-quit --logout")  # Example for GNOME; adjust as needed
        else:
            print("Logout not supported on this OS.")

    def analyze_frames(self, frames):
        """Analyze the frames to check for blinking, mouth movement, or head rotation."""
        if len(frames) < 2:
            return False  # Not enough frames to analyze

        alive = False
        blink_count = 0
        motion_detected = False
        previous_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        left_blinking = False
        right_blinking = False
        blink_frames = 0

        for i in range(1, len(frames)):
            current_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Calculate the difference between the current frame and the previous frame
            frame_diff = cv2.absdiff(previous_frame, current_frame)
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # Count non-zero pixels (indicating motion)
            non_zero_count = np.sum(thresh)
            if non_zero_count > 50000:  # Lowered threshold for smaller frames
                motion_detected = True

            previous_frame = current_frame  # Update previous frame for the next iteration

            # Detect faces in the current frame (use smaller upsized if needed, but keep small)
            faces = detector(current_frame, 1)  # Upsample=1 for speed

            if faces:
                shape = predictor(current_frame, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                # Get the eye landmarks
                left_eye = landmarks[36:42]  # Points 36-41 are the left eye
                right_eye = landmarks[42:48]  # Points 42-47 are the right eye

                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                if left_ear < 0.25:  # Threshold can be adjusted
                    left_blinking = True
                    blink_frames += 1  # Count frames where the left eye is detected as blinking

                if right_ear < 0.25:  # Threshold can be adjusted
                    right_blinking = True
                    blink_frames += 1  # Count frames where the right eye is detected as blinking

                if blink_frames >= 1:  # This could be adjusted
                    blink_count += 1
                    left_blinking = False  # Reset after counting
                    right_blinking = False

                if left_ear >= 0.25 and right_ear >= 0.25:
                    blink_frames = 0

        # Determine if the person is alive based on blink and motion detection
        if blink_count > 0 and motion_detected:
            alive = True

        return alive

    def eye_aspect_ratio(self, eye):
        """Calculate the Eye Aspect Ratio (EAR) for an eye given its landmarks."""
        A = distance.euclidean(eye[1], eye[5])  # 6th point - 2nd point
        B = distance.euclidean(eye[2], eye[4])  # 5th point - 3rd point
        C = distance.euclidean(eye[0], eye[3])  # 4th point - 1st point
        ear = (A + B) / (2.0 * C)
        return ear

if __name__ == "__main__":
    # Run as admin for USB control (check via ctypes)
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("Warning: Run as administrator for USB port control.")
    except:
        pass

    app = FaceRecognitionApp()
    try:
        app.run()
    except KeyboardInterrupt:
        app.close_app()
    finally:
        camera.__del__()
        if lock_window:
            lock_window.destroy()