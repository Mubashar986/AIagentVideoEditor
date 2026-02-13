# 🎬 YouTube Shorts AI Agent

An AI-powered pipeline that **downloads** a YouTube video, **transcribes** it,
uses an **LLM to pick the most engaging segments**, and **auto-edits** each
segment into a vertical short (9:16, ≤60s, with burned-in captions) — ready
for upload.

## Architecture

```
YouTube URL → Downloader (yt-dlp)
           → Transcriber (faster-whisper on GPU)
           → AI Analyzer (Llama 3 via Groq — free)
           → Video Editor (MoviePy + FFmpeg)
           → Upload-Ready Shorts (9:16, .mp4)
```

## Two Modes

| Mode | Transcription | LLM | Cost |
|------|--------------|-----|------|
| **`local`** (default) | faster-whisper (GPU) | Llama 3 via Groq | **$0** |
| **`openai`** | OpenAI Whisper API | GPT-4o-mini | ~$0.01/run |

## 🚀 Quick Start (Google Colab — FREE)

👉 **See [COLAB_GUIDE.md](COLAB_GUIDE.md)** for step-by-step Colab setup.

**TL;DR:**
1. Open [Google Colab](https://colab.research.google.com), enable **T4 GPU**
2. Get a free [Groq API key](https://console.groq.com) (no credit card)
3. Clone this repo, install deps, run `python app.py` → get a shareable web link

## 💻 Local Setup

```bash
cd AIagent
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # Add your GROQ_API_KEY
python main.py "https://youtu.be/VIDEO_ID" --shorts 3
```

## Web UI (Gradio)

```bash
python app.py
# Opens a web UI with a public shareable link
```

## Project Structure

```
AIagent/
├── main.py                     # CLI entry point
├── app.py                      # Gradio web frontend
├── COLAB_GUIDE.md              # Google Colab setup guide
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py               # Settings (MODE=local/openai)
│   ├── downloader.py           # YouTube download (yt-dlp)
│   ├── transcriber_local.py    # 🆓 faster-whisper (GPU)
│   ├── transcriber.py          # 💳 OpenAI Whisper API
│   ├── analyzer_local.py       # 🆓 Llama 3 via Groq
│   ├── analyzer.py             # 💳 GPT-4o-mini
│   ├── editor.py               # Video editor (MoviePy)
│   └── pipeline.py             # Orchestrator
├── downloads/                  # Source videos (git-ignored)
└── output/                     # Generated shorts (git-ignored)
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `MODE` | `local` | `local` (free) or `openai` (paid) |
| `GROQ_API_KEY` | — | Free key from console.groq.com |
| `WHISPER_LOCAL_MODEL` | `base` | tiny/base/small/medium/large-v3 |
| `SHORT_MAX_DURATION` | `55` | Max short duration (seconds) |

## License

For personal / educational use. Respect YouTube's Terms of Service.
