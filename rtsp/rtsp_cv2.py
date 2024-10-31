from flask import Flask, Response, request, abort, send_from_directory
import cv2
import ipaddress
import threading
import logging
import time
import queue

app = Flask(__name__, static_folder="public")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
    "door": "rtsp://admin:farahoosh@3207@172.16.28.6",   # RTSP camera
    "kouche": "rtsp://camera:FARAwallboard@192.168.1.212",  # RTSP camera
}

# Global variables for capturing frames for each camera
frame_queues = {camera_id: queue.Queue(maxsize=30) for camera_id in cameras.keys()}
capture_threads = {}

# Define the allowed IPs
allowed_ips = [
    '172.16.28.166',
    '127.0.0.1',
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

def capture_frames(camera_id, rtsp_source):
    cap = cv2.VideoCapture(rtsp_source)
    if not cap.isOpened():
        logging.error(f"Camera {camera_id} could not be opened.")
        return

    while True:
        success, frame = cap.read()
        if success:
            # Process the frame before putting it into the queue
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if ret:
                try:
                    # Put the processed frame into the queue without blocking
                    frame_queues[camera_id].put(buffer.tobytes(), block=True)
                except queue.Full:
                    logging.warning(f"Queue for {camera_id} is full, skipping frame...")
        else:
            logging.warning(f"Failed to read frame from {camera_id}. Reconnecting...")
            cap.release()
            time.sleep(1)  # Wait before trying to reconnect
            cap = cv2.VideoCapture(rtsp_source)  # Reconnect if failed

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in cameras:
        return "Camera not found", 404

    if camera_id not in capture_threads:
        capture_threads[camera_id] = threading.Thread(target=capture_frames, args=(camera_id, cameras[camera_id]), daemon=True)
        capture_threads[camera_id].start()

    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(camera_id):
    while True:
        try:
            frame = frame_queues[camera_id].get(timeout=1)  # Wait for a frame with timeout
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            continue

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
        if camera_id not in capture_threads:
            capture_threads[camera_id] = threading.Thread(target=capture_frames, args=(camera_id, rtsp_source), daemon=True)
            capture_threads[camera_id].start()

    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
