import cv2
import numpy as np
import pymongo
import base64
import threading
from PIL import Image, ImageTk
import tkinter as tk
import io
import time
import dlib
import face_recognition

# Load the pre-trained face landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://172.16.28.91:27017/')
db = client['face_recognition']
collection = db['users']

# Load known faces from MongoDB
def load_known_faces():
    faces = []
    names = []
    for entry in collection.find():
        name = entry['name']
        images = entry['images']
        for key in images.keys():
            image_data = base64.b64decode(images[key].split(",")[1])
            img = Image.open(io.BytesIO(image_data))
            img = np.array(img)  # Convert image to numpy array
            face_encoding = face_recognition.face_encodings(img)
            if face_encoding:
                faces.append(face_encoding[0])
                names.append(name)
    return faces, names

class VideoCamera:
    def __init__(self, source):
        self.video_source = source
        self.vid = cv2.VideoCapture(self.video_source)
        self.running = True
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

    def update_frame(self):
        while self.running:
            success, frame = self.vid.read()
            if success:
                self.frame = frame

    def get_frame(self):
        if hasattr(self, 'frame'):
            return self.frame
        else:
            return None

    def __del__(self):
        self.running = False
        self.vid.release()

# Create a global camera object
camera = VideoCamera(0)

# Load known faces
known_faces, known_names = load_known_faces()

# Create a GUI application
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")

        # Create a canvas for video display
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Create buttons
        self.capture_button = tk.Button(root, text="Capture and Detect", command=self.capture_and_detect)
        self.capture_button.pack()

        self.check_alive_button = tk.Button(root, text="Check for Signs of Life", command=self.check_alive)
        self.check_alive_button.pack()

        self.update()

    def update(self):
        frame = camera.get_frame()
        if frame is not None:
            # Resize the frame for display
            rgb_frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)

            # Convert to PhotoImage for tkinter
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

        self.root.after(10, self.update)

    def capture_and_detect(self):
        frame = camera.get_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Draw rectangles and labels for recognized faces
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display the captured frame with detections in a new window
            self.show_captured_image(frame)

    def show_captured_image(self, frame):
        # Create a new window
        result_window = tk.Toplevel(self.root)
        result_window.title("Captured Image with Detections")

        # Convert frame to Image for displaying
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        # Create a label to display the image
        label = tk.Label(result_window, image=imgtk)
        label.image = imgtk  # Keep a reference to avoid garbage collection
        label.pack()

    def check_alive(self):
        frames = []
        for _ in range(50):  # Capture 10 frames
            frame = camera.get_frame()
            if frame is not None:
                frames.append(frame.copy())  # Use copy to preserve the frame
                time.sleep(0.1)  # Wait for 0.5 seconds between captures for more frames

        # Analyze captured frames
        alive = self.analyze_frames(frames)
        if alive:
            result_message = "Person is alive (detected movement)."
        else:
            result_message = "Person is not alive (no movement detected)."

        # Display result in a new window
        self.show_result(result_message)

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
                print(f"Motion detected between frames {i-1} and {i}: {non_zero_count} pixels changed.")

            previous_frame = current_frame  # Update previous frame for the next iteration

            # Detect faces in the current frame
            faces = detector(current_frame)

            if faces:
                print(f"Detected {len(faces)} face(s) in frame {i}.")
                shape = predictor(current_frame, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                # Get the eye landmarks
                left_eye = landmarks[36:42]  # Points 36-41 are the left eye
                right_eye = landmarks[42:48]  # Points 42-47 are the right eye

                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                print(f"Left EAR: {left_ear}, Right EAR: {right_ear}")

                # Check if either eye is open (i.e., EAR above threshold)
                if left_ear < 0.25:  # Increased threshold
                    left_blinking = True
                    blink_frames += 1  # Count frames where the left eye is detected as blinking

                if right_ear < 0.25:  # Increased threshold
                    right_blinking = True
                    blink_frames += 1  # Count frames where the right eye is detected as blinking

                # If both eyes have been detected as blinking for a number of consecutive frames, we count it as a blink
                if blink_frames >= 1:  # This could be adjusted
                    blink_count += 1
                    print(f"Blink detected in frame {i}. Blink count: {blink_count}")
                    left_blinking = False  # Reset after counting
                    right_blinking = False

                # Reset the blink frame counter if neither eye is blinking
                if left_ear >= 0.25 and right_ear >= 0.25:
                    blink_frames = 0

        # Determine if the person is alive based on blink and motion detection
        if blink_count > 0 and motion_detected:
            alive = True

        print(f"Final assessment: {'Alive' if alive else 'Not alive'} (Blinks: {blink_count}, Motion: {motion_detected})")
        return alive

    def eye_aspect_ratio(self, eye):
        """Calculate the Eye Aspect Ratio (EAR) for a given eye."""
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def show_result(self, message):
        """Display the result in a new window."""
        result_window = tk.Toplevel(self.root)
        result_window.title("Alive Check Result")
        label = tk.Label(result_window, text=message)
        label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
