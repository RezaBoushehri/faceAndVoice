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
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Camera dictionary
cameras = {
    # "102": "rtsp://admin:farahoosh@3207@192.168.1.102",  # RTSP camera
    "entrance reception": "rtsp://admin:farahoosh@3207@192.168.1.201",  # RTSP entrance reception
    "Dedicated Unit": "rtsp://admin:farahoosh@3207@192.168.1.202",  # RTSP Dedicated Unit
    "Lobby 1st": "rtsp://admin:farahoosh@3207@192.168.1.203",  # RTSP lobby
    "Support Unit 1st": "rtsp://admin:farahoosh@3207@192.168.1.204",  # RTSP Support Unit 1st
    "Support Unit 2nd": "rtsp://admin:farahoosh@3207@192.168.1.205",  # RTSP Support Unit 2nd
    "Lobby 2nd": "rtsp://admin:farahoosh@3207@192.168.1.206",  # RTSP Lobby 2nd
    "Accountancy": "rtsp://admin:farahoosh@3207@192.168.1.207",  # RTSP Accountancy
    "Managers": "rtsp://admin:farahoosh@3207@192.168.1.208",  # RTSP Managers
    "conference": "rtsp://admin:farahoosh@3207@192.168.1.209",  # RTSP conference
    "Developers": "rtsp://admin:farahoosh@3207@192.168.1.211",  # RTSP Developers
    "Management secFloor": "rtsp://admin:farahoosh@3207@192.168.1.213",  # RTSP Management secFloor
    "Server FirstFloor": "rtsp://admin:farahoosh@3207@192.168.1.216",  # RTSP Server FirstFloor
    "Parking": "rtsp://admin:farahoosh@3207@192.168.1.217",  # RTSP Parking
    "storage": "rtsp://admin:farahoosh@3207@192.168.1.218",  # RTSP storage
    "kitchen": "rtsp://admin:farahoosh@3207@192.168.1.219",  # RTSP kitchen
    "door": "rtsp://admin:farahoosh@3207@172.16.28.6",   # RTSP camera
    "kouche": "rtsp://camera:FARAwallboard@192.168.1.212",  # RTSP camera
}


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

# Load known faces
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

# Camera handling class
class VideoCamera:
    def __init__(self, source):
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

# Main face recognition app class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Selection Face Recognition")

        # Dropdown menu for camera selection
        self.selected_camera = tk.StringVar(root)
        self.selected_camera.set("Select Camera")
        self.camera_options = tk.OptionMenu(root, self.selected_camera, *cameras.keys(), command=self.change_camera)
        self.camera_options.pack()

        # Canvas to display the selected camera feed
        self.canvas = tk.Canvas(root, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        self.canvas.pack()

        # Start button to initiate face recognition on selected camera
        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack()

        self.current_camera = None  # Track current camera instance

    def change_camera(self, selected_camera):
        """Change camera feed based on selection."""
        if self.current_camera:
            del self.current_camera  # Release previous camera if exists
        # Initialize a new VideoCamera instance for the selected camera
        camera_url = cameras[selected_camera]
        self.current_camera = VideoCamera(camera_url)
        self.update()

    def update(self):
        """Display the selected camera feed."""
        if self.current_camera:
            frame = self.current_camera.get_frame()
            if frame is not None:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.canvas.imgtk = imgtk

        # Continue updating every 33 ms
        self.root.after(33, self.update)

    def start_recognition(self):
        """Run face recognition on the current camera feed."""
        if self.current_camera:
            frame = self.current_camera.get_frame()
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.capture_and_detect(rgb_frame, self.selected_camera.get())

    def capture_and_detect(self, frame, camera_name):
        # Detect face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        detected_faces = set()  # To avoid duplicate logs for the same person in one frame

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if face encoding matches any known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"  # Default to Unknown if no match found

            # Determine the best match face
            if matches:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                else:
                    # Capture unknown faces in the database if not recognized
                    self.capture_unknown_face(frame[top:bottom, left:right])

            detected_faces.add(name)  # Add detected face name to the set

            # Draw rectangle around each face and add a label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Log each detected face
            print(f"Detected {name} in camera {camera_name}")

        # Display updated frame with drawn faces and labels
        self.display_frame_with_detections(frame)
    def display_frame_with_detections(self, frame):
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
        self.canvas.imgtk = imgtk

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

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
