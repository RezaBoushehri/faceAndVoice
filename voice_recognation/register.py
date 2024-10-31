import numpy as np
import librosa
from pymongo import MongoClient
from bson.binary import Binary
import sounddevice as sd
import base64
import logging
import wave
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import keyboard  # Add this import at the top

# Configure logging
logging.basicConfig(level=logging.INFO)

class SpeakerRecognition:
    def __init__(self, db):
        self.models = {}
        self.db = db
        self.scaler = StandardScaler()  # Feature scaler for MFCC normalization

    def save_audio_to_file(self, audio_binary, filename):
        """Saves the audio binary to a WAV file for debugging."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(44100)  # Sample rate
            wf.writeframes(audio_binary)

    def register_user(self, username):
        """Registers a new user by capturing their voice sample."""
        if self.db.voice_data.find_one({'username': username}):
            logging.info(f"Username {username} already exists. Choose a different name.")
            return False

        # Record user's voice for registration
        logging.info("Recording new user voice sample...")
        voice_sample = record_voice()
        
        # Save to database
        self.db.voice_data.insert_one({
            'username': username,
            'voice_sample': voice_sample
        })
        logging.info(f"User {username} registered successfully.")
        return True

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
            self.models[username] = self.scaler.fit_transform(mfcc_features)  # Normalize features

    def process_audio_base64(self, audio_base64, username):
        """Decodes Base64 audio and converts it to a NumPy array."""
        try:
            audio_binary = base64.b64decode(audio_base64)
            logging.info(f'User: {username}, Audio binary size: {len(audio_binary)} bytes')

            return np.frombuffer(audio_binary, dtype=np.int16)
        except Exception as e:
            logging.error(f"Error decoding audio for {username}: {e}")
            self.save_audio_to_file(audio_binary, f"error_{username}.wav")
            return None

    def extract_mfcc(self, audio_data, sr=44100, fixed_length=220500):
        """Extracts MFCC with chroma and spectral contrast features from audio data."""
        # Ensure audio is of the required length by padding or truncating
        if len(audio_data) < fixed_length:
            audio_data = np.pad(audio_data, (0, fixed_length - len(audio_data)), mode='constant')
        else:
            audio_data = audio_data[:fixed_length]

        # Convert int16 audio data to float32 format
        audio_data = audio_data.astype(np.float32) / 32768.0  # Scale to range [-1, 1]

        # Extract MFCCs and delta features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine MFCC and delta features
        combined_mfcc = np.vstack((mfcc, mfcc_delta, mfcc_delta2))

        # Extract chroma and spectral contrast features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)

        # Stack all features for a comprehensive feature set
        features = np.vstack((combined_mfcc, chroma, spectral_contrast))
        return features.T
    def predict(self, audio_data):
        """Predicts the speaker from the input audio data using cosine similarity."""
        input_features = self.extract_mfcc(audio_data)
        input_features = self.scaler.transform(input_features)  # Normalize the input features
        input_mean = np.mean(input_features, axis=0)

        scores = {}
        for username, model_features in self.models.items():
            model_mean = np.mean(model_features, axis=0)
            distance = cosine(model_mean, input_mean)
            scores[username] = distance

        # Identify the speaker with the lowest cosine distance
        if scores:
            recognized_user = min(scores, key=scores.get)
            logging.info(f"Scores: {scores}")
            return recognized_user if scores[recognized_user] < 1.2 else "Unknown"  # Adjusted threshold
        return "Unknown"

def record_voice():
    """Records voice while holding down the space bar and encodes it to Base64."""
    fs = 44100  # Sample rate
    duration = 0  # Initialize duration
    recording_list = []
    
    logging.info("Press and hold the space bar to start recording...")

    while True:
        if keyboard.is_pressed('space'):
            # Record in small chunks to build up duration
            chunk = sd.rec(int(0.1 * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            recording_list.append(chunk.flatten())  # Add each chunk to the list
            
        else:
            if recording_list:  # Stop if there’s a recording to process
                logging.info("Finished recording.")
                break
    
    # Concatenate all chunks into a single array
    recording = np.concatenate(recording_list)
    
    # Encode binary audio as Base64
    audio_binary = recording.tobytes()
    audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
    return audio_base64
                         
def recognize_speaker():
    """Main function to recognize the speaker."""
    client = MongoClient('mongodb://172.16.28.200:27017/')
    db = client.voice_db

    recognizer = SpeakerRecognition(db)
    recognizer.load_voice_data()

    recorded_audio_base64 = record_voice()
    recorded_audio_data = base64.b64decode(recorded_audio_base64)
    recorded_audio_array = np.frombuffer(recorded_audio_data, dtype=np.int16)

    recognized_user = recognizer.predict(recorded_audio_array)
    logging.info(f'Recognized User: {recognized_user}')

def register_new_user(username):
    """Wrapper function to register a new user."""
    client = MongoClient('mongodb://172.16.28.200:27017/')
    db = client.voice_db

    recognizer = SpeakerRecognition(db)
    recognizer.register_user(username)

if __name__ == "__main__":
    # Example usage
    register_new_user("Mr.Jafari")
    # recognize_speaker()
                                                   