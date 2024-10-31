from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import base64
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client.voice_db
collection = db.voice_data

@app.route('/')
def index():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    audio_data = request.files.get('audio')
    username = request.form.get('username')

    if not audio_data or not username:
        logging.error('Missing audio data or username.')
        return jsonify({"message": "Missing audio data or username."}), 400

    try:
        # Read audio data and save it as a Base64 encoded string in MongoDB
        audio_binary = audio_data.read()  # Read the full binary content
        audio_base64 = base64.b64encode(audio_binary).decode('utf-8')  # Encode to Base64 and decode to string

        logging.info(f'Received audio data for user: {username}, Size: {len(audio_binary)} bytes')

        # Insert into MongoDB
        collection.insert_one({'username': username, 'voice_sample': audio_base64})
        logging.info(f'User {username} registered successfully. Voice sample size: {len(audio_base64)} characters')
        
        return jsonify({"message": f"User {username} registered successfully."}), 201

    except Exception as e:
        logging.error(f'Error registering user {username}: {e}')
        return jsonify({"message": f"Error registering user {username}."}), 500

if __name__ == '__main__':
    app.run(debug=True)
