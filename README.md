# PC Audio AI Assistant

A local-first intelligent assistant that listens to all system audio (e.g., Teams, WhatsApp, Slack),
transcribes it in real time, detects questions asked by others, and provides AI-generated responses
based on user context using Retrieval-Augmented Generation (RAG).

## Features

- System-wide audio capture (WASAPI - Windows)
- Real-time speech-to-text with Faster-Whisper
- Speaker diarization with user voice profile
- Keyword-based question detection
- RAG pipeline with local vector store (FAISS/Chroma)
- Local or remote LLM via Ollama or OpenAI
- UI overlay using Tauri
- Multilingual architecture (English, Spanish, French, Portuguese, German)

## Requirements

- Python 3.10+
- Rust (for Tauri UI)
- Node.js (for Tauri front-end)

## Getting Started

```bash
git clone https://github.com/grcamauer/pc-audio-ai-assistant.git
cd pc-audio-ai-assistant
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## License

MIT
