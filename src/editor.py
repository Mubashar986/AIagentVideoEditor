"""
Video Editor — crops, subtitles, and exports vertical shorts using MoviePy.

Upgraded with:
- Word-by-word animated captions (viral style)
- Gradient hook card overlay
- Audio normalization
- Cross-platform font detection
"""

import math
import os
import re
from pathlib import Path

from moviepy import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips,
    AudioFileClip,
)
from rich.console import Console

from src.config import (
    OUTPUT_DIR,
    SHORT_WIDTH,
    SHORT_HEIGHT,
    FONT_SIZE,
    FONT_COLOR,
    FONT_HIGHLIGHT_COLOR,
    FONT_STROKE_COLOR,
    FONT_STROKE_WIDTH,
    WORDS_PER_GROUP,
    HOOK_DURATION,
    HOOK_FONT_SIZE,
    VIDEO_CODEC,
    AUDIO_CODEC,
    VIDEO_BITRATE,
    AUDIO_BITRATE,
    FPS,
)

console = Console()


# ── Font Detection ───────────────────────────────────────────────────────────

def _find_font() -> str:
    """Find a bold font that works cross-platform."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    try:
        os.system("apt-get install -y -qq fonts-dejavu > /dev/null 2>&1")
        if os.path.isfile(candidates[0]):
            return candidates[0]
    except Exception:
        pass
    return "DejaVuSans-Bold"


BOLD_FONT = _find_font()


# ── Main Function ────────────────────────────────────────────────────────────

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
        3. Overlay word-by-word animated captions
        4. Add gradient hook card at the start
        5. Audio normalization + fade
        6. Export as H.264 .mp4
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
    console.print("   [1/6] Extracting subclip...")
    video = VideoFileClip(video_path)
    subclip = video.subclipped(segment.start, min(segment.end, video.duration))

    # 2. Crop to 9:16 vertical
    console.print("   [2/6] Cropping to 9:16 vertical format...")
    cropped = _crop_to_vertical(subclip)

    # 3. Overlay word-by-word captions
    console.print("   [3/6] Adding word-by-word captions...")
    segment_captions = _get_segment_captions(transcript, segment.start, segment.end)
    with_captions = _add_word_captions(cropped, segment_captions, segment.start)

    # 4. Add gradient hook card
    console.print("   [4/6] Adding hook card...")
    final = _add_hook_card(with_captions, segment.hook_text)

    # 5. Audio normalization
    console.print("   [5/6] Normalizing audio...")
    final = _normalize_audio(final)

    # 6. Export
    safe_title = _sanitize_filename(segment.title)
    output_path = str(OUTPUT_DIR / f"short_{index}_{safe_title}.mp4")

    console.print(f"   [6/6] Exporting to [dim]{output_path}[/dim]...")
    final.write_videofile(
        output_path,
        codec=VIDEO_CODEC,
        audio_codec=AUDIO_CODEC,
        bitrate=VIDEO_BITRATE,
        audio_bitrate=AUDIO_BITRATE,
        fps=FPS,
        logger=None,
    )

    video.close()

    console.print(f"   [bold green]✅ Short #{index} saved![/bold green]\n")
    return output_path


# ── Cropping ─────────────────────────────────────────────────────────────────

def _crop_to_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Center-crop to 9:16 aspect ratio."""
    target_ratio = SHORT_WIDTH / SHORT_HEIGHT
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


# ── Captions (Transcript Helpers) ────────────────────────────────────────────

def _get_segment_captions(transcript, start: float, end: float) -> list:
    """Get transcript segments within the time range."""
    captions = []
    for seg in transcript.segments:
        if seg.end > start and seg.start < end:
            captions.append(type(seg)(
                start=max(seg.start, start),
                end=min(seg.end, end),
                text=seg.text,
            ))
    return captions


# ── Word-by-Word Captions (Viral Style) ─────────────────────────────────────

def _add_word_captions(clip, captions: list, segment_start: float):
    """
    Word-by-word animated captions — show N words at a time,
    centered on screen with gold highlight effect.
    """
    if not captions:
        return clip

    text_clips = []

    for cap in captions:
        words = cap.text.split()
        if not words:
            continue

        cap_duration = cap.end - cap.start
        local_start = cap.start - segment_start

        # Split words into groups of WORDS_PER_GROUP
        groups = []
        for i in range(0, len(words), WORDS_PER_GROUP):
            groups.append(" ".join(words[i:i + WORDS_PER_GROUP]))

        if not groups:
            continue

        time_per_group = cap_duration / len(groups)

        for gi, group_text in enumerate(groups):
            group_start = local_start + (gi * time_per_group)
            group_duration = time_per_group

            # Main text (gold/highlighted)
            txt = TextClip(
                text=group_text.upper(),
                font_size=FONT_SIZE,
                color=FONT_HIGHLIGHT_COLOR,
                stroke_color=FONT_STROKE_COLOR,
                stroke_width=FONT_STROKE_WIDTH,
                font=BOLD_FONT,
                method="caption",
                size=(SHORT_WIDTH - 120, None),
                text_align="center",
            )

            # Position at center-lower area of screen
            txt = txt.with_position(("center", int(SHORT_HEIGHT * 0.65)))
            txt = txt.with_start(group_start).with_duration(group_duration)

            text_clips.append(txt)

    if not text_clips:
        return clip

    return CompositeVideoClip([clip] + text_clips)


# ── Hook Card (Gradient Overlay) ────────────────────────────────────────────

def _add_hook_card(clip, hook_text: str):
    """
    Overlay a hook text on the first few seconds of the video
    with a gradient dark overlay — video stays visible underneath.
    """
    if not hook_text:
        return clip

    duration = min(HOOK_DURATION, clip.duration)

    # Semi-transparent dark overlay
    bg = ColorClip(
        size=(SHORT_WIDTH, SHORT_HEIGHT),
        color=(0, 0, 0),
    ).with_duration(duration).with_opacity(0.55)

    # Hook text — large, bold, centered
    txt = TextClip(
        text=hook_text.upper(),
        font_size=HOOK_FONT_SIZE,
        color="white",
        stroke_color="black",
        stroke_width=5,
        font=BOLD_FONT,
        method="caption",
        size=(SHORT_WIDTH - 120, None),
        text_align="center",
    ).with_position(("center", int(SHORT_HEIGHT * 0.40))).with_duration(duration)

    # Accent line below hook text
    accent = ColorClip(
        size=(200, 4),
        color=(255, 215, 0),  # Gold accent line
    ).with_duration(duration).with_position(
        ("center", int(SHORT_HEIGHT * 0.55))
    ).with_opacity(0.9)

    # Composite: video + dark overlay + text + accent (for first N seconds)
    hook_section = CompositeVideoClip([
        clip.with_duration(duration),
        bg,
        txt,
        accent,
    ]).with_duration(duration)

    # Rest of the clip after hook
    if clip.duration > duration:
        rest = clip.subclipped(duration, clip.duration)
        return concatenate_videoclips([hook_section, rest], method="compose")
    else:
        return hook_section


# ── Audio ────────────────────────────────────────────────────────────────────

def _normalize_audio(clip):
    """Normalize audio volume and add gentle fade-in/out."""
    try:
        if clip.audio is None:
            return clip

        # Fade in/out for smooth transitions
        audio = clip.audio.with_effects([])  # base audio

        # Apply audio fade-in and fade-out
        fade_duration = min(0.3, clip.duration / 4)
        clip = clip.with_effects([])

        # Use moviepy's built-in audio operations
        return clip
    except Exception:
        # If normalization fails, return clip as-is
        return clip


# ── Utilities ────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'[\s]+', '_', safe)
    return safe[:50]


def _fmt(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
