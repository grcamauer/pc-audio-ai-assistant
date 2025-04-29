# transcribe_with_whispercpp.py

import whispercpp
import numpy as np
import scipy.io.wavfile as wav
import os

AUDIO_FILE = "mic_test.wav"
MODEL_NAME = "small.en"  # options: tiny, base, small, medium, large

def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    rate, data = wav.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]  # Take one channel if stereo
    data = data.astype(np.float32) / np.iinfo(data.dtype).max  # Normalize
    return rate, data

def main():
    print(f"Loading model: {MODEL_NAME}")
    model = whispercpp.Whisper.from_pretrained(MODEL_NAME)

    print(f"Loading audio: {AUDIO_FILE}")
    sample_rate, audio = load_audio(AUDIO_FILE)

    print("Running transcription...")
    model.transcribe(audio, sample_rate=sample_rate)
    segments = model.segments()

    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    main()
