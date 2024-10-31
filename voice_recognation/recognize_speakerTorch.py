import numpy as np
import torch
from pymongo import MongoClient
import base64
import logging
import sounddevice as sd
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)

class SpeakerRecognitionSystem:
    model = None  # Class attribute to store the model for all instances

    def __init__(self, db):
        self.db = db
        self.models = {}  # Initialize the dictionary to store user embeddings
        self.audio_cache = {}  # Cache to store processed audio for each user
        if SpeakerRecognitionSystem.model is None:
            SpeakerRecognitionSystem.model = self.load_model()  # Load model only once
        
    def load_model(self):
        """Loads the pre-trained SpeakerRecognition model from SpeechBrain if not already loaded."""
        logging.info("Loading pre-trained speaker recognition model...")
        try:
            model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir_asv")
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    def process_audio_base64(self, audio_base64, username):
        """Decodes Base64 audio and converts it to a NumPy array, only if not already processed."""
        if username in self.audio_cache:
            logging.info(f"Using cached audio data for user: {username}")
            return self.audio_cache[username]

        try:
            # Decode Base64 to bytes
            audio_binary = base64.b64decode(audio_base64)
            logging.info(f'User: {username}, Audio binary size: {len(audio_binary)} bytes')

            # Convert binary audio to a 1D NumPy array
            audio_data = np.frombuffer(audio_binary, dtype=np.int16)

            # Cache the audio data for future use
            self.audio_cache[username] = audio_data
            return audio_data
        except Exception as e:
            logging.error(f"Error decoding audio for {username}: {e}")
            # Save for debugging if decoding fails
            with open(f"error_{username}.wav", "wb") as f:
                f.write(base64.b64decode(audio_base64))  # Save the original Base64 audio
            return None
    def extract_embedding(self, audio_data):
        """Extracts speaker embedding from audio data using SpeechBrain."""
        audio_tensor = torch.tensor(audio_data).float().unsqueeze(0)  # Add batch dimension
        output = SpeakerRecognitionSystem.model(audio_tensor)  # Use the class model

        # Check if the output is a tuple
        if isinstance(output, tuple):
            embedding = output[0]  # Usually, the first element is the embedding
        else:
            embedding = output  # If not a tuple, use the output directly

        # Print the shape of the resulting embedding for debugging
        logging.info(f"Extracted embedding shape: {embedding.shape}")

        # Ensure that the embedding is a 1D tensor
        return embedding.squeeze() if embedding.dim() > 1 else embedding


    def load_voice_data(self):
        """Loads and decodes Base64-encoded voice samples from the database."""
        users = self.db.voice_data.find()
        for user in users:
            username = user['username']
            audio_base64 = user['voice_sample']  # This is a Base64 string
            audio_data = self.process_audio_base64(audio_base64, username)

            if audio_data is None:
                continue

            embedding = self.extract_embedding(audio_data)
            self.models[username] = embedding

    def predict(self, audio_data, k=3):
        """Predicts the speaker based on the recorded audio data."""
        recorded_embedding = self.extract_embedding(audio_data)

        # Collect distances and usernames
        distances = []
        for username, stored_embedding in self.models.items():
            distance = cosine(recorded_embedding.detach().numpy(), stored_embedding.detach().numpy())
            distances.append((distance, username))
        
        # Sort distances and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:k]

        # Log distances for debugging
        for distance, username in nearest_neighbors:
            logging.info(f"Distance to {username}: {distance}")

        # Determine the most common label among the nearest neighbors
        most_common_username = Counter(username for _, username in nearest_neighbors).most_common(1)

        # If we have no common user, we might want to return "Unknown"
        if not most_common_username:
            return "Unknown"

        recognized_user = most_common_username[0][0]  # Get the most common username
        recognized_distance = nearest_neighbors[0][0]  # Get the distance of the most common user

        # Define a threshold for acceptance (adjust this based on your model's performance)
        threshold = 0.6  # Adjust this based on your model's performance
        if recognized_distance < threshold:  # Ensure we're comparing a float distance
            return recognized_user
        
        return "Unknown"  # If the distance is above the threshold
    
def record_voice(duration=5):
    """Records voice for a specified duration and encodes it to Base64."""
    fs = 44100  # Sample rate
    logging.info("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    logging.info("Finished recording.")
    audio_binary = recording.flatten().tobytes()  # Convert to binary

    # Encode binary audio as Base64
    audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
    return audio_base64

def recognize_speaker():
    """Main function to recognize the speaker."""
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://172.16.28.200:27017/')
        db = client.voice_db

        recognizer = SpeakerRecognitionSystem(db)
        recognizer.load_voice_data()  # Load existing voice samples

        # Record audio and encode to Base64
        recorded_audio_base64 = record_voice()
        recorded_audio_data = base64.b64decode(recorded_audio_base64)
        recorded_audio_array = np.frombuffer(recorded_audio_data, dtype=np.int16)

        # Predict the speaker
        recognized_user = recognizer.predict(recorded_audio_array)
        logging.info(f"user speach : {recognized_user}")
        return recognized_user
    except Exception as e:
        logging.error(f"Error in speaker recognition: {e}")

if __name__ == "__main__":
    recognize_speaker()
