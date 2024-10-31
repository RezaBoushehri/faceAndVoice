import numpy as np
import librosa
from pymongo import MongoClient
from bson.binary import Binary
import sounddevice as sd
import base64
import logging
import wave

# Configure logging
logging.basicConfig(level=logging.INFO)

class SpeakerRecognition:
    def __init__(self, db):
        self.models = {}
        self.db = db

    def save_audio_to_file(self, audio_binary, filename):
        """Saves the audio binary to a WAV file for debugging."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(44100)  # Sample rate
            wf.writeframes(audio_binary)

    def load_voice_data(self):
        """Loads and decodes Base64-encoded voice samples from the database."""
        users = self.db.voice_data.find()
        for user in users:
            username = user['username']
            audio_base64 = user['voice_sample']  # This is a Base64 string
            audio_data = self.process_audio_base64(audio_base64, username)

            if audio_data is None:
                continue

            mfcc_features = self.extract_mfcc(audio_data)
            self.models[username] = mfcc_features

    def process_audio_base64(self, audio_base64, username):
        """Decodes Base64 audio and converts it to a NumPy array."""
        try:
            # Decode Base64 to bytes
            audio_binary = base64.b64decode(audio_base64)
            logging.info(f'User: {username}, Audio binary size: {len(audio_binary)} bytes')

            # Convert binary audio to a 1D NumPy array
            return np.frombuffer(audio_binary, dtype=np.int16)
        except Exception as e:
            logging.error(f"Error decoding audio for {username}: {e}")
            self.save_audio_to_file(audio_binary, f"error_{username}.wav")  # Save for debugging
            return None

    def extract_mfcc(self, audio_data):
        """Extracts MFCC features from audio data."""
        sr = 44100  # Sample rate for librosa
        mfcc = librosa.feature.mfcc(y=audio_data.astype(np.float32), sr=sr, n_mfcc=40)
        return mfcc.T

    def predict(self, audio_data):
        """Predicts the speaker from the input audio data."""
        input_mfcc = self.extract_mfcc(audio_data)
        input_mfcc_mean = np.mean(input_mfcc, axis=0, keepdims=True)

        scores = {}
        for username, model_features in self.models.items():
            model_mfcc_mean = np.mean(model_features, axis=0, keepdims=True)
            distance = np.mean((model_mfcc_mean - input_mfcc_mean) ** 2)
            scores[username] = distance

        if scores:
            recognized_user = min(scores, key=scores.get)
            logging.info(f"Scores: {scores}")
            return recognized_user if scores[recognized_user] < 20 else "Unknown"
        return "Unknown"  # No users in the model

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
    # Connect to MongoDB
    client = MongoClient('mongodb://172.16.28.200:27017/')
    db = client.voice_db

    recognizer = SpeakerRecognition(db)
    recognizer.load_voice_data()  # Load existing voice samples

    # Record audio and encode to Base64
    recorded_audio_base64 = record_voice()
    recorded_audio_data = base64.b64decode(recorded_audio_base64)  # Decode Base64 to binary for prediction
    recorded_audio_array = np.frombuffer(recorded_audio_data, dtype=np.int16)

    # Predict the speaker
    recognized_user = recognizer.predict(recorded_audio_array)
    logging.info(f'Recognized User: {recognized_user}')

if __name__ == "__main__":
    recognize_speaker()
