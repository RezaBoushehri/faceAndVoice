import tkinter as tk
from tkinter import messagebox
import wave
import pyaudio
from recognize_speaker import recognize_speaker

class VoiceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recognition App")

        self.label = tk.Label(root, text="Press 'Record' to speak.")
        self.label.pack()

        self.record_button = tk.Button(root, text="Record", command=self.record_voice)
        self.record_button.pack()

        self.recognize_button = tk.Button(root, text="Recognize", command=self.recognize_voice)
        self.recognize_button.pack()

    def record_voice(self):
        filename = "temp_voice.wav"
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 5

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        messagebox.showinfo("Info", "Recording saved as temp_voice.wav")

    def recognize_voice(self):
        recognized_user = recognize_speaker("temp_voice.wav")
        messagebox.showinfo("Recognition Result", f"Recognized User: {recognized_user}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecognitionApp(root)
    root.mainloop()
