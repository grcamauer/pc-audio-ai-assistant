# capture.py

import sounddevice as sd
import numpy as np
import asyncio
from typing import AsyncGenerator

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

def find_device(name_keywords):
    for i, device in enumerate(sd.query_devices()):
        if all(kw.lower() in device['name'].lower() for kw in name_keywords):
            return i
    return None

async def audio_stream_loopback() -> AsyncGenerator[np.ndarray, None]:
    """Capture system audio using WASAPI loopback on Windows."""
    loopback_device = find_device(['loopback'])
    if loopback_device is None:
        raise RuntimeError("Loopback device not found. Ensure WASAPI loopback is supported and enabled.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=loopback_device,
        blocksize=BLOCK_SIZE
    )

    with stream:
        while True:
            data, _ = stream.read(BLOCK_SIZE)
            yield data
            await asyncio.sleep(0)

async def audio_stream_microphone() -> AsyncGenerator[np.ndarray, None]:
    """Capture audio from the default microphone."""
    mic_device = sd.default.device[0]  # Default input device
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=mic_device,
        blocksize=BLOCK_SIZE
    )

    with stream:
        while True:
            data, _ = stream.read(BLOCK_SIZE)
            yield data
            await asyncio.sleep(0)
