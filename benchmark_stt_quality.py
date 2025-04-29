# benchmark_stt_quality.py

from faster_whisper import WhisperModel
import time
import scipy.io.wavfile as wav
import numpy as np
import os

AUDIO_FILE = "mic_test.wav"
MODELS = ["tiny", "base", "small", "medium"]

def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    rate, data = wav.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]  # Take first channel if stereo
    data = data.astype(np.float32) / np.iinfo(data.dtype).max  # Normalize
    return rate, data

def benchmark(model_size, compute_type="int8"):
    print(f"Benchmarking model: {model_size} ({compute_type})")
    model = WhisperModel(model_size, compute_type=compute_type)

    sample_rate, audio = load_audio(AUDIO_FILE)
    duration = len(audio) / sample_rate

    start = time.time()
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
    end = time.time()

    elapsed = end - start
    rtf = elapsed / duration

    print(f"⏱ Time: {elapsed:.2f}s for {duration:.2f}s of audio")
    print(f"⚡ Real-Time Factor (RTF): {rtf:.2f}x")
    print("📝 Transcription:")
    for seg in segments:
        print(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
    print("-" * 60)

if __name__ == "__main__":
    for model in MODELS:
        try:
            benchmark(model)
        except Exception as e:
            print(f"❌ Error with model {model}: {e}")
