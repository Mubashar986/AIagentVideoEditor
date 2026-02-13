"""
Centralized configuration for the YouTube Shorts AI Agent.
Supports two modes: 'openai' (paid) and 'local' (free, for Colab).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Mode ─────────────────────────────────────────────────────────────────────
# Set to "local" for free open-source models (Colab), "openai" for paid APIs
MODE = os.getenv("MODE", "local")

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────────
# OpenAI (only needed if MODE="openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Groq (free — get key at https://console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Settings ───────────────────────────────────────────────────────────
# OpenAI models (paid mode)
WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o-mini"

# Local models (free mode / Colab)
WHISPER_LOCAL_MODEL = os.getenv("WHISPER_LOCAL_MODEL", "base")  # tiny/base/small/medium/large-v3
GROQ_MODEL = "llama-3.1-8b-instant"

# ── Video / Shorts Settings ─────────────────────────────────────────────────
SHORT_MAX_DURATION = 55
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920

# ── Caption / Subtitle Settings ─────────────────────────────────────────────
FONT_SIZE = 60
FONT_COLOR = "white"
FONT_STROKE_COLOR = "black"
FONT_STROKE_WIDTH = 3
CAPTION_POSITION = ("center", 0.75)

# ── Export Settings ──────────────────────────────────────────────────────────
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_BITRATE = "5000k"
AUDIO_BITRATE = "192k"
FPS = 30
