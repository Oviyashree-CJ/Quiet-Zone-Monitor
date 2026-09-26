from audio_processing.feature_extraction import extract_features
from mqtt_client.mqtt_publisher import send_result
import sounddevice as sd
from scipy.io.wavfile import write

import joblib
import time
import numpy as np 

fs = 16000        # sampling rate
duration = 2      # seconds
model = joblib.load("ml_model/human_sound_model.pkl")
scaler = joblib.load("ml_model/scaler.pkl")

def get_sound_level(audio):
    rms = np.sqrt(np.mean(audio**2))
    return rms

while True:
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    #  SOUND LEVEL (no file needed)
    sound_level = get_sound_level(audio)
    print("Sound Level:", sound_level)

    #  ONLY IF ML NEEDS FILE
    write("input.wav", fs, audio)

    features = extract_features("input.wav")
    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    msg = f"{prediction},{sound_level}"

    print("Prediction:", prediction)
    send_result(msg)

    time.sleep(5)