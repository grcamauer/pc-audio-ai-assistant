# record_test.py

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

DURATION = 15  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1
FILENAME = "mic_test.wav"

print(f"Recording microphone for {DURATION} seconds...")

recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
sd.wait()

wav.write(FILENAME, SAMPLE_RATE, recording)
print(f"Saved to {FILENAME}")
