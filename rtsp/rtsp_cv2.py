from flask import Flask, Response, request, abort, send_from_directory
import cv2
import ipaddress
import threading
import time
import logging
import numpy as np

app = Flask(__name__, static_folder="public")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Global variables for capturing frames for each camera
frame_buffers = {}
frame_locks = {}
capture_threads = {}
camera_statuses = {}  # New dictionary to track camera statuses

# Define the allowed IPs
allowed_ips = [
    '172.16.28.166',  # CIDR GOD
    '127.0.0.1',      # CIDR mr.Goudarzi
]

def allowed_ip():
    remote_ip = request.remote_addr
    for allowed in allowed_ips:
        if '/' in allowed:
            if ipaddress.ip_address(remote_ip) in ipaddress.ip_network(allowed):
                return
        elif remote_ip == allowed:
            return
    abort(403)  # Forbidden if the IP is not allowed

@app.before_request
def before_request():
    allowed_ip()

# Replace with your RTSP URLs for multiple cameras
cameras = {
    "102": "rtsp://admin:farahoosh@3207@192.168.1.102",  # RTSP camera
    "201": "rtsp://admin:farahoosh@3207@192.168.1.201",  # RTSP entrance reception
    "202": "rtsp://admin:farahoosh@3207@192.168.1.202",  # RTSP Dedicated Unit
    "203": "rtsp://admin:farahoosh@3207@192.168.1.203",  # RTSP camera
    "204": "rtsp://admin:farahoosh@3207@192.168.1.204",  # RTSP Support Unit 1st
    "205": "rtsp://admin:farahoosh@3207@192.168.1.205",  # RTSP Support Unit 2nd
    "206": "rtsp://admin:farahoosh@3207@192.168.1.206",  # RTSP Lobby 2nd
    "207": "rtsp://admin:farahoosh@3207@192.168.1.207",  # RTSP Accountancy
    "208": "rtsp://admin:farahoosh@3207@192.168.1.208",  # RTSP Managers
    "209": "rtsp://admin:farahoosh@3207@192.168.1.209",  # RTSP conference
    "211": "rtsp://admin:farahoosh@3207@192.168.1.211",  # RTSP Developers
    "213": "rtsp://admin:farahoosh@3207@192.168.1.213",  # RTSP Management secFloor
    "216": "rtsp://admin:farahoosh@3207@192.168.1.216",  # RTSP Server FirstFloor
    "217": "rtsp://admin:farahoosh@3207@192.168.1.217",  # RTSP Parking
    "218": "rtsp://admin:farahoosh@3207@192.168.1.218",  # RTSP camera
    "219": "rtsp://admin:farahoosh@3207@192.168.1.219",  # RTSP kitchen
    "door": "rtsp://admin:farahoosh@3207@172.16.28.6",  # RTSP camera
    "kouche": "rtsp://camera:FARAwallboard@192.168.1.212",  # RTSP camera
}

def capture_frames(camera_id, rtsp_source):
    global frame_buffers, frame_locks

    frame_buffers[camera_id] = None
    frame_locks[camera_id] = threading.Lock()
    cap = cv2.VideoCapture(rtsp_source)

    if not cap.isOpened():
        logging.error(f"Unable to open video source {rtsp_source} for {camera_id}. Retrying...")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)  # Increase buffer size if necessary

    while True:
        if not cap.isOpened():
            logging.warning(f"Reconnecting to {camera_id}...")
            cap.open(rtsp_source)  # Attempt reconnection

        success, frame = cap.read()
        if not success:
            logging.warning(f"Failed to read frame from {camera_id}, retrying in 1 second...")
            time.sleep(1)
            continue

        try:
            # Resize the frame
            frame = cv2.resize(frame, (640, 480))

            # Acquire lock to update frame buffer
            with frame_locks[camera_id]:
                frame_buffers[camera_id] = frame

        except cv2.error as e:
            logging.error(f"OpenCV error for {camera_id}: {e}")
            time.sleep(1)  # Sleep to avoid tight loop on error

    cap.release()



def generate_frames(camera_id):
    global frame_buffers, frame_locks
    while True:
        with frame_locks[camera_id]:
            if frame_buffers[camera_id] is None:
                logging.debug(f"No frames available for {camera_id}, sleeping...")
                time.sleep(0.1)  # Avoid high CPU load in case of no frames
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, buffer = cv2.imencode('.jpg', frame_buffers[camera_id], encode_param)
            if not ret:
                logging.warning(f"Failed to encode frame for {camera_id}, skipping...")
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.1)  # Slight delay to control CPU usage

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in cameras:
        return "Camera not found", 404

    # Ensure only one thread per camera
    if camera_id not in capture_threads:
        capture_threads[camera_id] = threading.Thread(target=capture_frames, args=(camera_id, cameras[camera_id]), daemon=True)
        capture_threads[camera_id].start()

    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html')

@app.route('/css/<file>')
def serve_css(file):
    return send_from_directory('css', file)

@app.route('/js/<file>')
def serve_js(file):
    return send_from_directory('js', file)

if __name__ == "__main__":
    # Start capturing frames for each camera at the application startup
    for camera_id, rtsp_source in cameras.items():
        capture_threads[camera_id] = threading.Thread(target=capture_frames, args=(camera_id, rtsp_source), daemon=True)
        capture_threads[camera_id].start()

    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
