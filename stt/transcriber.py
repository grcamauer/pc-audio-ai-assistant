# transcriber.py

import os
import numpy as np
from faster_whisper import WhisperModel

# Dynamically set threading environment variables based on CPU count
cpu_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

# Load Whisper model (adjust model size and compute_type as needed)
model = WhisperModel("small.en", compute_type="int8")

def transcribe_audio_chunk(chunk: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe a chunk of audio (numpy array) and return the text."""
    if chunk.ndim > 1:
        chunk = chunk[:, 0]

    segments, _ = model.transcribe(
        chunk,
        language="en",
        beam_size=3,
        vad_filter=True
    )
    return " ".join(segment.text for segment in segments).strip()
