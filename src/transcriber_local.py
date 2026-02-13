"""
Local Transcriber — uses faster-whisper (CTranslate2) for GPU-accelerated
transcription. Free, no API key needed.
"""

from dataclasses import dataclass, field
import tempfile
from pathlib import Path

from moviepy import VideoFileClip
from rich.console import Console

console = Console()


@dataclass
class TranscriptSegment:
    """A single segment of the transcript."""
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """Full transcript with segments."""
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0


def transcribe(video_path: str, model_size: str = "base") -> Transcript:
    """
    Transcribe a video using faster-whisper running locally on GPU.

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size — 'tiny', 'base', 'small', 'medium', 'large-v3'.
                    'base' is good balance of speed vs accuracy for Colab T4.

    Returns:
        Transcript with full_text and per-segment timestamps.
    """
    from faster_whisper import WhisperModel

    console.print(f"\n[bold cyan]🎙️  Transcribing audio (faster-whisper, model={model_size})...[/bold cyan]\n")

    # Step 1: Extract audio
    audio_path = _extract_audio(video_path)

    # Step 2: Load model on GPU
    console.print(f"   Loading Whisper model '{model_size}' on GPU...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Step 3: Transcribe
    console.print("   Transcribing...")
    segments_gen, info = model.transcribe(audio_path, beam_size=5)

    # Step 4: Collect segments
    segments = []
    full_text_parts = []
    for seg in segments_gen:
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        ))
        full_text_parts.append(seg.text.strip())

    transcript = Transcript(
        full_text=" ".join(full_text_parts),
        segments=segments,
        language=info.language,
        duration=segments[-1].end if segments else 0.0,
    )

    # Cleanup
    Path(audio_path).unlink(missing_ok=True)

    console.print(f"[bold green]✅ Transcription complete![/bold green]")
    console.print(f"   Language:  [white]{transcript.language}[/white]")
    console.print(f"   Segments:  [white]{len(transcript.segments)}[/white]")
    console.print(f"   Full text: [dim]{transcript.full_text[:120]}...[/dim]\n")

    return transcript


def _extract_audio(video_path: str) -> str:
    """Extract audio track to a temporary mp3 file."""
    console.print("   Extracting audio track from video...")

    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, logger=None)
    video.close()

    console.print(f"   Audio extracted → [dim]{temp_audio_path}[/dim]")
    return temp_audio_path
