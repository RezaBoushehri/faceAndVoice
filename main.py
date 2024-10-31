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
import dlib
from scipy.spatial import distance
from voice_recognation.recognize_speakerTorch import recognize_speaker

# Define frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Initialize HOG descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

# MongoDB connection
client = pymongo.MongoClient('mongodb://172.16.28.91:27017/')
db = client['face_recognition']
collection = db['users']

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
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                self.frame = frame
            time.sleep(0.03)

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

        # Add start recognition button
        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack()

        self.face_detected = False
        self.update()

    def update(self):
        frame = camera.get_frame()
        if frame is not None:
            # Display live feed
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.canvas.imgtk = imgtk

        # Keep updating every 33 ms
        self.root.after(33, self.update)

    def start_recognition(self):
        frame = camera.get_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.capture_and_detect(rgb_frame)

    def capture_and_detect(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
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

        # Capture frames for movement analysis if a human is detected
        frames = [camera.get_frame().copy() for _ in range(30) if camera.get_frame() is not None]
        for name in detected_faces:
            if recognize_speaker() == name:
                alive = self.analyze_frames(frames)
                self.log_alive_status(alive, name)

    def log_alive_status(self, alive, name):
        status = "alive" if alive else "not alive"
        print(f"Logged: {name} is {status} at {datetime.now()}")

    def analyze_frames(self, frames):
        alive = False
        blink_count = 0
        motion_detected = False
        previous_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        blink_frames = 0

        for i in range(1, len(frames)):
            current_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(previous_frame, current_frame)
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            non_zero_count = np.sum(thresh)
            if non_zero_count > 10000:
                motion_detected = True
            previous_frame = current_frame

            faces = detector(current_frame)
            if faces:
                shape = predictor(current_frame, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                if left_ear < 0.25 or right_ear < 0.25:
                    blink_frames += 1
                if blink_frames >= 1:
                    blink_count += 1
                    blink_frames = 0

        if blink_count > 0 and motion_detected:
            alive = True

        return alive

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
