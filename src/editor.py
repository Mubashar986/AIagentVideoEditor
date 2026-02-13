"""
Video Editor — crops, subtitles, and exports vertical shorts using MoviePy.
Works with both local and OpenAI mode transcripts/segments (duck-typed).
"""

import os
import re
from pathlib import Path

from moviepy import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips,
)
from PIL import ImageFont
from rich.console import Console

from src.config import (
    OUTPUT_DIR,
    SHORT_WIDTH,
    SHORT_HEIGHT,
    FONT_SIZE,
    FONT_COLOR,
    FONT_STROKE_COLOR,
    FONT_STROKE_WIDTH,
    VIDEO_CODEC,
    AUDIO_CODEC,
    VIDEO_BITRATE,
    AUDIO_BITRATE,
    FPS,
)

console = Console()


def _find_font() -> str:
    """Find a bold font that works cross-platform (Windows, Linux/Colab, Mac)."""
    candidates = [
        # Linux / Colab
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        # Mac
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Last resort: try to install fonts on Linux (Colab)
    try:
        os.system("apt-get install -y -qq fonts-dejavu > /dev/null 2>&1")
        if os.path.isfile(candidates[0]):
            return candidates[0]
    except Exception:
        pass

    # Fallback — let Pillow try to resolve it
    return "DejaVuSans-Bold"


BOLD_FONT = _find_font()


def create_short(
    video_path: str,
    segment,
    transcript,
    index: int = 1,
) -> str:
    """
    Create a single YouTube Short from a video segment.

    Pipeline:
        1. Extract subclip for the segment time range
        2. Crop/resize to 9:16 vertical format
        3. Overlay animated captions from transcript
        4. Add a hook-text title card at the start
        5. Export as H.264 .mp4

    Args:
        video_path: Path to the source video.
        segment: Object with start, end, title, hook_text attributes.
        transcript: Object with segments list (each having start, end, text).
        index: Short number (for output filename).

    Returns:
        Path to the exported short .mp4 file.
    """
    console.print(
        f"\n[bold cyan]🎬 Creating Short #{index}: "
        f"[white]{segment.title}[/white][/bold cyan]"
    )
    console.print(
        f"   Time: {_fmt(segment.start)} → {_fmt(segment.end)} "
        f"({segment.end - segment.start:.0f}s)\n"
    )

    # 1. Load and subclip
    console.print("   [1/5] Extracting subclip...")
    video = VideoFileClip(video_path)
    subclip = video.subclipped(segment.start, min(segment.end, video.duration))

    # 2. Crop to 9:16 vertical
    console.print("   [2/5] Cropping to 9:16 vertical format...")
    cropped = _crop_to_vertical(subclip)

    # 3. Overlay captions
    console.print("   [3/5] Adding captions...")
    segment_captions = _get_segment_captions(transcript, segment.start, segment.end)
    with_captions = _add_captions(cropped, segment_captions, segment.start)

    # 4. Add hook title card
    console.print("   [4/5] Adding hook title card...")
    final = _add_hook_card(with_captions, segment.hook_text)

    # 5. Export
    safe_title = _sanitize_filename(segment.title)
    output_path = str(OUTPUT_DIR / f"short_{index}_{safe_title}.mp4")

    console.print(f"   [5/5] Exporting to [dim]{output_path}[/dim]...")
    final.write_videofile(
        output_path,
        codec=VIDEO_CODEC,
        audio_codec=AUDIO_CODEC,
        bitrate=VIDEO_BITRATE,
        audio_bitrate=AUDIO_BITRATE,
        fps=FPS,
        logger=None,
    )

    # Cleanup
    video.close()

    console.print(f"   [bold green]✅ Short #{index} saved![/bold green]\n")
    return output_path


# ── Internal Helpers ─────────────────────────────────────────────────────────


def _crop_to_vertical(clip: VideoFileClip) -> VideoFileClip:
    """
    Center-crop a clip to 9:16 aspect ratio.
    """
    target_ratio = SHORT_WIDTH / SHORT_HEIGHT  # 0.5625
    src_w, src_h = clip.size
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        x_offset = (src_w - new_w) // 2
        cropped = clip.cropped(x1=x_offset, x2=x_offset + new_w)
    else:
        new_h = int(src_w / target_ratio)
        y_offset = (src_h - new_h) // 2
        cropped = clip.cropped(y1=y_offset, y2=y_offset + new_h)

    return cropped.resized((SHORT_WIDTH, SHORT_HEIGHT))


def _get_segment_captions(transcript, start: float, end: float) -> list:
    """Get transcript segments that fall within the time range."""
    captions = []
    for seg in transcript.segments:
        if seg.end > start and seg.start < end:
            captions.append(type(seg)(
                start=max(seg.start, start),
                end=min(seg.end, end),
                text=seg.text,
            ))
    return captions


def _add_captions(clip, captions: list, segment_start: float):
    """
    Overlay subtitle text clips on the video.
    """
    if not captions:
        return clip

    text_clips = []
    for cap in captions:
        txt = TextClip(
            text=cap.text,
            font_size=FONT_SIZE,
            color=FONT_COLOR,
            stroke_color=FONT_STROKE_COLOR,
            stroke_width=FONT_STROKE_WIDTH,
            font=BOLD_FONT,
            method="caption",
            size=(SHORT_WIDTH - 80, None),
            text_align="center",
        )

        txt = txt.with_position(("center", int(SHORT_HEIGHT * 0.72)))

        local_start = cap.start - segment_start
        local_end = cap.end - segment_start
        txt = txt.with_start(local_start).with_duration(local_end - local_start)

        text_clips.append(txt)

    return CompositeVideoClip([clip] + text_clips)


def _add_hook_card(clip, hook_text: str, duration: float = 2.0):
    """
    Prepend a short title card with the hook text.
    """
    if not hook_text:
        return clip

    bg = ColorClip(
        size=(SHORT_WIDTH, SHORT_HEIGHT),
        color=(0, 0, 0),
    ).with_duration(duration).with_opacity(0.7)

    txt = TextClip(
        text=hook_text,
        font_size=FONT_SIZE + 10,
        color="white",
        font=BOLD_FONT,
        method="caption",
        size=(SHORT_WIDTH - 100, None),
        text_align="center",
    ).with_position("center").with_duration(duration)

    title_card = CompositeVideoClip([
        clip.with_duration(duration),
        bg,
        txt,
    ]).with_duration(duration)

    final = concatenate_videoclips([title_card, clip], method="compose")
    return final


def _sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'[\s]+', '_', safe)
    return safe[:50]


def _fmt(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
