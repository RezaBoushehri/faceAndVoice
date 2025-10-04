import cv2
import numpy as np
import pymongo
import base64
import threading
from PIL import Image, ImageTk
import tkinter as tk
import io
import face_recognition
from datetime import datetime
import time
import dlib  # Make sure to install dlib
from scipy.spatial import distance  # For EAR calculation
import pyttsx3  # For text-to-speech; install with: pip install pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from voice_recognation.recognize_speakerTorch import recognize_speaker
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
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.canvas = tk.Canvas(root, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        self.canvas.pack()
        
        self.detecting = False
        self.face_detected = False
        self.update()

    def update(self):
        frame = camera.get_frame()
        if frame is not None and not self.face_detected:
            # Reduce frame resolution to 640x480 for lower resource usage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect only if human is detected and face not logged recently
            if self.detect_human(rgb_frame):
                self.capture_and_detect(rgb_frame)

            # Show frame on the canvas
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(33, self.update)  # Increase update interval to 33 ms

    def detect_human(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8))
        return len(humans) > 0

    def capture_and_detect(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Prepare to log detected faces
        detected_faces = set()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if matches:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    detected_faces.add(name)
                else:
                    detected_faces.add('Unknown')

            # Draw rectangles and labels
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ** Capture frames for movement analysis if a human is detected **
        frames = []
        for _ in range(30):  # Capture 10 frames
            captured_frame = camera.get_frame()
            if captured_frame is not None:
                frames.append(captured_frame.copy())  # Use copy to preserve the frame
                time.sleep(0.03)  # Wait for 0.03 seconds between captures

        # Log detected faces
        for name in detected_faces:
            alive = self.analyze_frames(frames)
            self.log_alive_status(alive, name, frames)

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
        
        if alive:
            if name != "BB":
                # Save picture in magels folder with timestamp
                if frames:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{timestamp_str}.jpg"
                    os.makedirs("magels", exist_ok=True)
                    cv2.imwrite(f"magels/{filename}", frames[-1])
                self.send_email_alert(name)
                self.logout_system()
            else:
                # For BB, perform voice recognition
                recognized_voice = recognize_speaker()
                if recognized_voice == name:
                    self.play_tts("Welcome BB, access granted.")
                    self.close_app()
                else:
                    # Save suspicious BB picture
                    if frames:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"suspicious_BB_{timestamp_str}.jpg"
                        os.makedirs("magels", exist_ok=True)
                        cv2.imwrite(f"magels/{filename}", frames[-1])
                    self.send_email_alert(name)
                    self.logout_system()

    def close_app(self):
        self.root.quit()
        self.root.destroy()

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

            body = f"Unauthorized person detected: {name}. Timestamp: {datetime.now()}. Action: Logging out."
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
            if non_zero_count > 10000:  # Threshold can be adjusted
                motion_detected = True

            previous_frame = current_frame  # Update previous frame for the next iteration

            # Detect faces in the current frame
            faces = detector(current_frame)

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
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()