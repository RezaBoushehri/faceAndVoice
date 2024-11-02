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

cameras = {
    "102": "rtsp://admin:farahoosh@192.168.1.102",
    "201": "rtsp://admin:farahoosh@192.168.1.201",
    "202": "rtsp://admin:farahoosh@192.168.1.202",
    "203": "rtsp://admin:farahoosh@192.168.1.203",
    "204": "rtsp://admin:farahoosh@192.168.1.204",
    "205": "rtsp://admin:farahoosh@192.168.1.205",
    "206": "rtsp://admin:farahoosh@192.168.1.206",
    "207": "rtsp://admin:farahoosh@192.168.1.207",
    "208": "rtsp://admin:farahoosh@192.168.1.208",
    "209": "rtsp://admin:farahoosh@192.168.1.209",
    "211": "rtsp://admin:farahoosh@192.168.1.211",
    "213": "rtsp://admin:farahoosh@192.168.1.213",
    "216": "rtsp://admin:farahoosh@192.168.1.216",
    "217": "rtsp://admin:farahoosh@192.168.1.217",
    "218": "rtsp://admin:farahoosh@192.168.1.218",
    "219": "rtsp://admin:farahoosh@192.168.1.219",
    "door": "rtsp://admin:farahoosh@172.16.28.6",
    "kouche": "rtsp://camera:FARAwallboard@192.168.1.212",
}

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
    def __init__(self, source="rtsp://admin:farahoosh@192.168.1.211"):
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
        self.thread.join()
        self.vid.release()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.canvas = tk.Canvas(root, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        self.canvas.pack()

        # Dropdown for camera selection
        self.selected_camera = tk.StringVar(root)
        self.selected_camera.set(list(cameras.keys())[0])  # Set default camera

        camera_menu = tk.OptionMenu(root, self.selected_camera, *cameras.keys())
        camera_menu.pack()

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
        # Get the selected camera URL
        selected_camera_key = self.selected_camera.get()
        camera_url = cameras[selected_camera_key]

        # Reinitialize the camera with the selected RTSP URL
        global camera
        camera.__del__()  # Release current camera
        camera = VideoCamera(source=camera_url)

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

            # Capture unknown face and log it
            if name == "Unknown":
                self.capture_unknown_face(frame[top:bottom, left:right])

        # Capture frames for movement analysis if a human is detected
        frames = [camera.get_frame().copy() for _ in range(30) if camera.get_frame() is not None]
        for name in detected_faces:
            print(f"Logged: {name} Detected")
            alive = self.analyze_frames(frames)
            if name != "Unknown":
                if recognize_speaker() == name:
                    self.log_alive_status(alive, name)
            else:
                self.log_alive_status(alive, name)

    def capture_unknown_face(self, face_image):
        """Encodes unknown face in Base64 and logs it with the date."""
        try:
            # Encode the face image to Base64
            _, buffer = cv2.imencode('.jpg', face_image)
            face_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare document with date and Base64 image
            unknown_log = {
                "date": datetime.now().isoformat(),
                "image": face_base64
            }

            # Insert into unknown_logs collection
            db.unknown_logs.insert_one(unknown_log)
            print("Unknown face logged in database.")
        except Exception as e:
            print(f"Error capturing unknown face: {e}")

    def log_alive_status(self, alive, name):
        status = "alive" if alive else "not alive"
        print(f"Logged: {name} is {status} at {datetime.now()}")

    def analyze_frames(self, frames):
        alive = False
        blink_count = 0
        motion_detected = False
        previous_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            delta_frame = cv2.absdiff(previous_frame, gray_frame)
            thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                motion_detected = True

            if blink_count >= 3:
                alive = True
            previous_frame = gray_frame

        return alive or motion_detected


if __name__ == "__main__":
    global camera
    camera = VideoCamera()  # Initialize with a default camera
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
