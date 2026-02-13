"""
Audio Transcriber — extracts audio from video and transcribes via OpenAI Whisper API.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from moviepy import VideoFileClip
from openai import OpenAI
from rich.console import Console

from src.config import OPENAI_API_KEY, WHISPER_MODEL

console = Console()


@dataclass
class TranscriptSegment:
    """A single segment of the transcript."""
    start: float   # seconds
    end: float     # seconds
    text: str


@dataclass
class Transcript:
    """Full transcript with segments."""
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0


def transcribe(video_path: str) -> Transcript:
    """
    Transcribe a video file using OpenAI Whisper API.

    1. Extracts audio track from the video.
    2. Sends audio to Whisper API.
    3. Returns structured transcript with timestamps.

    Args:
        video_path: Path to the video file.

    Returns:
        Transcript object with full_text and per-segment timestamps.
    """
    console.print(f"\n[bold cyan]🎙️  Transcribing audio...[/bold cyan]\n")

    # Step 1: Extract audio to a temporary WAV file
    audio_path = _extract_audio(video_path)

    # Step 2: Send to Whisper API
    console.print("   Sending audio to Whisper API...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # Step 3: Parse response into our Transcript model
    segments = []
    for seg in response.segments:
        segments.append(TranscriptSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
        ))

    transcript = Transcript(
        full_text=response.text,
        segments=segments,
        language=response.language,
        duration=segments[-1].end if segments else 0.0,
    )

    # Cleanup temp audio file
    Path(audio_path).unlink(missing_ok=True)

    console.print(f"[bold green]✅ Transcription complete![/bold green]")
    console.print(f"   Language:  [white]{transcript.language}[/white]")
    console.print(f"   Segments:  [white]{len(transcript.segments)}[/white]")
    console.print(f"   Full text: [dim]{transcript.full_text[:120]}...[/dim]\n")

    return transcript


def _extract_audio(video_path: str) -> str:
    """
    Extract the audio track from a video file to a temporary mp3 file.

    Returns:
        Path to the temporary audio file.
    """
    console.print("   Extracting audio track from video...")

    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, logger=None)
    video.close()

    console.print(f"   Audio extracted → [dim]{temp_audio_path}[/dim]")
    return temp_audio_path
