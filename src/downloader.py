"""
YouTube Video Downloader — wraps yt-dlp to download videos.
"""

import yt_dlp
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console

from src.config import DOWNLOADS_DIR

console = Console()


@dataclass
class DownloadResult:
    """Result of a video download."""
    video_path: str
    title: str
    duration: float  # seconds
    thumbnail_url: str


def download_video(url: str) -> DownloadResult:
    """
    Download a YouTube video in the best available mp4 quality.

    Args:
        url: YouTube video URL

    Returns:
        DownloadResult with path, title, duration, and thumbnail URL
    """
    console.print(f"\n[bold cyan]📥 Downloading video...[/bold cyan]")
    console.print(f"   URL: [dim]{url}[/dim]\n")

    # yt-dlp options
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info first to get metadata
        info = ydl.extract_info(url, download=True)

        title = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        thumbnail = info.get("thumbnail", "")

        # Determine the actual downloaded file path
        # yt-dlp sanitizes filenames, so we use the prepared filename
        video_path = ydl.prepare_filename(info)

        # Ensure the extension is .mp4 (yt-dlp may merge into mp4)
        video_path = str(Path(video_path).with_suffix(".mp4"))

    console.print(f"\n[bold green]✅ Download complete![/bold green]")
    console.print(f"   Title:    [white]{title}[/white]")
    console.print(f"   Duration: [white]{_format_duration(duration)}[/white]")
    console.print(f"   Saved to: [dim]{video_path}[/dim]\n")

    return DownloadResult(
        video_path=video_path,
        title=title,
        duration=duration,
        thumbnail_url=thumbnail,
    )


def _progress_hook(d: dict):
    """Progress callback for yt-dlp."""
    if d["status"] == "downloading":
        percent = d.get("_percent_str", "?%")
        speed = d.get("_speed_str", "?")
        eta = d.get("_eta_str", "?")
        console.print(
            f"   ⬇ {percent} | Speed: {speed} | ETA: {eta}",
            end="\r",
        )
    elif d["status"] == "finished":
        console.print(f"\n   [dim]Merging audio + video...[/dim]")


def _format_duration(seconds: float) -> str:
    """Format seconds into mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
