# main.py

import asyncio
import numpy as np
from audio.capture import audio_stream_microphone
from stt.transcriber import transcribe_audio_chunk

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
BUFFER_SECONDS = 5
OVERLAP_SECONDS = 1

BUFFER_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_SECONDS

async def main():
    print("Starting rolling-buffer transcription (press Ctrl+C to stop)...")
    last_text = ""
    
    mic_gen = audio_stream_microphone()
    buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)

    try:
        while True:
            mic_audio = await anext(mic_gen)

            if mic_audio.ndim > 1:
                mic_audio = mic_audio[:, 0]

            buffer = np.roll(buffer, -CHUNK_SIZE)
            buffer[-CHUNK_SIZE:] = mic_audio

            if np.any(buffer[:BUFFER_SAMPLES - OVERLAP_SAMPLES]):
                text = transcribe_audio_chunk(buffer)
                if text and text != last_text:
                    print("🗣️", text)
                    last_text = text

    except KeyboardInterrupt:
        print("Transcription stopped.")

if __name__ == "__main__":
    asyncio.run(main())
