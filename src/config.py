"""
Centralized configuration for the YouTube Shorts AI Agent.
Supports two modes: 'openai' (paid) and 'local' (free, for Colab).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Mode ─────────────────────────────────────────────────────────────────────
MODE = os.getenv("MODE", "local")

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"
THUMBNAILS_DIR = BASE_DIR / "thumbnails"

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Settings ───────────────────────────────────────────────────────────
WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o-mini"
WHISPER_LOCAL_MODEL = os.getenv("WHISPER_LOCAL_MODEL", "base")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Video / Shorts Settings ─────────────────────────────────────────────────
SHORT_MIN_DURATION = 25
SHORT_MAX_DURATION = 55
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920

# ── Caption Styles ──────────────────────────────────────────────────────────
# Available: "hormozi", "beast", "subtle", "karaoke"
CAPTION_STYLE = os.getenv("CAPTION_STYLE", "hormozi")

CAPTION_STYLES = {
    "hormozi": {
        "font_size": 80,
        "color": "white",
        "highlight_color": "#FFD700",
        "stroke_color": "black",
        "stroke_width": 5,
        "bg_enabled": True,
        "bg_color": (0, 0, 0),
        "bg_opacity": 0.6,
        "words_per_group": 3,
        "uppercase": True,
        "position_y": 0.72,
    },
    "beast": {
        "font_size": 85,
        "color": "white",
        "highlight_color": "#FF4444",
        "stroke_color": "black",
        "stroke_width": 6,
        "bg_enabled": True,
        "bg_color": (0, 0, 0),
        "bg_opacity": 0.7,
        "words_per_group": 2,
        "uppercase": True,
        "position_y": 0.70,
    },
    "subtle": {
        "font_size": 50,
        "color": "white",
        "highlight_color": "white",
        "stroke_color": "black",
        "stroke_width": 3,
        "bg_enabled": True,
        "bg_color": (0, 0, 0),
        "bg_opacity": 0.5,
        "words_per_group": 6,
        "uppercase": False,
        "position_y": 0.82,
    },
    "karaoke": {
        "font_size": 70,
        "color": "#AAAAAA",
        "highlight_color": "#FFFFFF",
        "stroke_color": "black",
        "stroke_width": 4,
        "bg_enabled": True,
        "bg_color": (0, 0, 0),
        "bg_opacity": 0.55,
        "words_per_group": 4,
        "uppercase": False,
        "position_y": 0.75,
    },
}

# ── Zoom Effect Settings ────────────────────────────────────────────────────
ZOOM_ENABLED = True
ZOOM_START = 1.0
ZOOM_END = 1.12

# ── Hook Card Settings ──────────────────────────────────────────────────────
HOOK_DURATION = 1.5
HOOK_FONT_SIZE = 80

# ── Export Settings ──────────────────────────────────────────────────────────
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_BITRATE = "5000k"
AUDIO_BITRATE = "192k"
FPS = 30
