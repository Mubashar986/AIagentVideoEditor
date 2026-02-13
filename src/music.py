"""
Background Music — download music from URL and mix with video audio.

Supports:
- YouTube URLs (via yt-dlp)
- Direct MP3/WAV URLs
- Local file paths
"""

import os
import tempfile
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"
MUSIC_DIR.mkdir(exist_ok=True)


def download_music(url: str) -> str:
    """
    Download audio from a URL (YouTube, SoundCloud, or direct link).

    Returns:
        Path to the downloaded .mp3 file.
    """
    # If it's already a local file
    if os.path.isfile(url):
        console.print(f"   🎵 Using local file: {url}")
        return url

    console.print(f"   🎵 Downloading music from URL...")

    output_path = str(MUSIC_DIR / "bg_music.mp3")

    # Use yt-dlp for YouTube/SoundCloud/etc.
    try:
        import yt_dlp

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(MUSIC_DIR / "bg_music.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        console.print(f"   ✅ Music downloaded → {output_path}")
        return output_path

    except Exception as e:
        console.print(f"   [yellow]⚠ yt-dlp failed ({e}), trying direct download...[/yellow]")

    # Fallback: direct download for MP3 URLs
    try:
        import urllib.request
        urllib.request.urlretrieve(url, output_path)
        console.print(f"   ✅ Music downloaded → {output_path}")
        return output_path
    except Exception as e:
        console.print(f"   [red]❌ Could not download music: {e}[/red]")
        return ""


def mix_audio(video_clip, music_path: str, music_volume: float = 0.15):
    """
    Mix background music with video's original audio.

    Args:
        video_clip: MoviePy VideoFileClip with audio.
        music_path: Path to background music file (.mp3/.wav).
        music_volume: Volume of background music (0.0 to 1.0).
                      Default 0.15 = 15% volume (subtle background).

    Returns:
        VideoFileClip with mixed audio.
    """
    try:
        from moviepy import AudioFileClip, CompositeAudioClip

        if not music_path or not os.path.isfile(music_path):
            return video_clip

        if video_clip.audio is None:
            return video_clip

        console.print(f"   🎵 Mixing background music (volume: {int(music_volume * 100)}%)...")

        # Load music
        music = AudioFileClip(music_path)

        # Loop or trim music to match video duration
        video_duration = video_clip.duration

        if music.duration < video_duration:
            # Loop the music to fill the video
            loops_needed = int(video_duration / music.duration) + 1
            from moviepy import concatenate_audioclips
            music_clips = [music] * loops_needed
            music = concatenate_audioclips(music_clips)

        # Trim to video length
        music = music.subclipped(0, video_duration)

        # Lower music volume
        music = music.with_volume_scaled(music_volume)

        # Add fade-in and fade-out to music
        fade_duration = min(1.0, video_duration / 4)
        music = music.audio_fadein(fade_duration).audio_fadeout(fade_duration)

        # Mix original audio + music
        original_audio = video_clip.audio
        mixed = CompositeAudioClip([original_audio, music])

        # Apply mixed audio to video
        result = video_clip.with_audio(mixed)

        console.print("   ✅ Background music added!")
        return result

    except Exception as e:
        console.print(f"   [yellow]⚠ Music mixing failed: {e}. Using original audio.[/yellow]")
        return video_clip
