"""
Video Editor — Premium quality shorts with:
- Caption style presets (hormozi/beast/subtle/karaoke)
- Dark background bars behind text (always readable)
- Ken Burns zoom effect
- Gradient hook card
- Thumbnail generator
- Cross-platform font detection
"""

import os
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from moviepy import (
    VideoFileClip,
    TextClip,
    ImageClip,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips,
)
from rich.console import Console

from src.config import (
    OUTPUT_DIR,
    THUMBNAILS_DIR,
    SHORT_WIDTH,
    SHORT_HEIGHT,
    CAPTION_STYLE,
    CAPTION_STYLES,
    ZOOM_ENABLED,
    ZOOM_START,
    ZOOM_END,
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
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            console.print(f"   [dim]Font: {path}[/dim]")
            return path

    # Try installing fonts on Linux/Colab
    try:
        os.system("apt-get install -y -qq fonts-dejavu > /dev/null 2>&1")
        if os.path.isfile(candidates[0]):
            return candidates[0]
    except Exception:
        pass

    return "Helvetica-Bold"


BOLD_FONT = _find_font()


def _get_style(style_name: str = "") -> dict:
    """Get caption style config, defaulting to CAPTION_STYLE."""
    name = style_name or CAPTION_STYLE
    return CAPTION_STYLES.get(name, CAPTION_STYLES["hormozi"])


# ── PIL-Based Text Rendering ────────────────────────────────────────────────

def _render_text_to_image(text: str, font_size: int, color: str,
                           stroke_color: str, stroke_width: int,
                           max_width: int) -> np.ndarray:
    """
    Render text to a NumPy array using PIL (100% reliable, no MoviePy bugs).
    Returns RGBA numpy array.
    """
    try:
        font = ImageFont.truetype(BOLD_FONT, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure text size
    dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + stroke_width * 2 + 20
    text_h = bbox[3] - bbox[1] + stroke_width * 2 + 20

    # Create transparent image
    img = Image.new("RGBA", (max_width, text_h + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Center text
    x = (max_width - (bbox[2] - bbox[0])) // 2
    y = 5

    # Draw stroke (outline)
    if stroke_width > 0:
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)

    # Draw main text
    draw.text((x, y), text, font=font, fill=color)

    return np.array(img)


# ── Main Function ────────────────────────────────────────────────────────────

def create_short(
    video_path: str,
    segment,
    transcript,
    index: int = 1,
    caption_style: str = "",
    music_path: str = "",
) -> str:
    """
    Create a single YouTube Short from a video segment.
    """
    style = _get_style(caption_style)

    console.print(
        f"\n[bold cyan]🎬 Creating Short #{index}: "
        f"[white]{segment.title}[/white][/bold cyan]"
    )
    console.print(
        f"   Time: {_fmt(segment.start)} → {_fmt(segment.end)} "
        f"({segment.end - segment.start:.0f}s)"
    )
    console.print(f"   Style: [bold]{caption_style or CAPTION_STYLE}[/bold]")
    if music_path:
        console.print(f"   Music: [bold]✅ enabled[/bold]")
    console.print()

    # 1. Load and subclip
    console.print("   [1/7] Extracting subclip...")
    video = VideoFileClip(video_path)
    subclip = video.subclipped(segment.start, min(segment.end, video.duration))

    # 2. Crop to 9:16 vertical + zoom
    console.print("   [2/7] Cropping to 9:16 + zoom effect...")
    cropped = _crop_to_vertical(subclip)
    if ZOOM_ENABLED:
        cropped = _apply_zoom(cropped)

    # 3. Word-by-word captions with background
    console.print("   [3/7] Adding captions...")
    segment_captions = _get_segment_captions(transcript, segment.start, segment.end)
    console.print(f"   [dim]Found {len(segment_captions)} caption segments[/dim]")
    with_captions = _add_styled_captions(cropped, segment_captions, segment.start, style)

    # 4. Hook card overlay
    console.print("   [4/7] Adding hook card...")
    with_hook = _add_hook_card(with_captions, segment.hook_text)

    # 5. Background music
    if music_path:
        console.print("   [5/7] Mixing background music...")
        from src.music import mix_audio
        final = mix_audio(with_hook, music_path, music_volume=0.15)
    else:
        console.print("   [5/7] No background music")
        final = with_hook

    # 6. Export
    safe_title = _sanitize_filename(segment.title)
    output_path = str(OUTPUT_DIR / f"short_{index}_{safe_title}.mp4")

    console.print(f"   [6/7] Exporting to [dim]{output_path}[/dim]...")
    final.write_videofile(
        output_path,
        codec=VIDEO_CODEC,
        audio_codec=AUDIO_CODEC,
        bitrate=VIDEO_BITRATE,
        audio_bitrate=AUDIO_BITRATE,
        fps=FPS,
        logger=None,
    )

    # 7. Generate thumbnail
    console.print("   [7/7] Generating thumbnail...")
    thumb_path = _generate_thumbnail(
        video_path, segment, safe_title, index
    )

    video.close()

    console.print(f"   [bold green]✅ Short #{index} saved![/bold green]")
    if thumb_path:
        console.print(f"   🖼️  Thumbnail: [dim]{thumb_path}[/dim]\n")
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


# ── Zoom Effect (Ken Burns) ─────────────────────────────────────────────────

def _apply_zoom(clip):
    """Apply slow zoom-in (Ken Burns) for cinematic feel."""
    duration = clip.duration

    def zoom_frame(get_frame, t):
        progress = t / duration if duration > 0 else 0
        scale = ZOOM_START + (ZOOM_END - ZOOM_START) * progress

        frame = get_frame(t)
        h, w = frame.shape[:2]

        new_w = int(w / scale)
        new_h = int(h / scale)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        cropped = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cropped)
        pil_img = pil_img.resize((w, h), PILImage.LANCZOS)
        return np.array(pil_img)

    return clip.transform(zoom_frame)


# ── Captions ─────────────────────────────────────────────────────────────────

def _get_segment_captions(transcript, start: float, end: float) -> list[dict]:
    """
    Get transcript segments within the time range.
    Returns simple dicts to avoid type() reconstruction bugs.
    """
    captions = []
    for seg in transcript.segments:
        if seg.end > start and seg.start < end:
            captions.append({
                "start": max(seg.start, start),
                "end": min(seg.end, end),
                "text": seg.text.strip(),
            })
    return captions


def _add_styled_captions(clip, captions: list[dict], segment_start: float, style: dict):
    """
    Word-by-word captions using PIL rendering (reliable on all platforms).
    """
    if not captions:
        return clip

    overlay_clips = []
    font_size = style["font_size"]
    color = style["highlight_color"]
    stroke_color = style["stroke_color"]
    stroke_width = style["stroke_width"]
    words_per = style["words_per_group"]
    uppercase = style["uppercase"]
    pos_y = int(SHORT_HEIGHT * style["position_y"])
    bg_enabled = style["bg_enabled"]

    for cap in captions:
        text = cap["text"]
        words = text.split()
        if not words:
            continue

        cap_duration = cap["end"] - cap["start"]
        local_start = cap["start"] - segment_start

        # Ensure positive timing
        if local_start < 0:
            local_start = 0
        if cap_duration <= 0:
            continue

        # Split into word groups
        groups = []
        for i in range(0, len(words), words_per):
            groups.append(" ".join(words[i:i + words_per]))

        if not groups:
            continue

        time_per_group = cap_duration / len(groups)

        for gi, group_text in enumerate(groups):
            group_start = local_start + (gi * time_per_group)
            group_duration = time_per_group

            if group_duration < 0.05:
                continue

            display_text = group_text.upper() if uppercase else group_text

            if not display_text.strip():
                continue

            # Dark background bar
            if bg_enabled:
                bg_clip = _create_caption_background(
                    font_size, style, group_start, group_duration
                )
                if bg_clip:
                    overlay_clips.append(bg_clip)

            # Render text using PIL (reliable!)
            text_img = _render_text_to_image(
                display_text, font_size, color, stroke_color,
                stroke_width, SHORT_WIDTH - 80
            )

            txt_clip = (
                ImageClip(text_img)
                .with_position(("center", pos_y))
                .with_start(group_start)
                .with_duration(group_duration)
            )
            overlay_clips.append(txt_clip)

    if not overlay_clips:
        return clip

    return CompositeVideoClip([clip] + overlay_clips)


def _create_caption_background(font_size: int, style: dict,
                                start: float, duration: float):
    """Create a semi-transparent dark bar behind caption text."""
    try:
        bar_height = font_size + 60
        pos_y = int(SHORT_HEIGHT * style["position_y"]) - 20

        bg = ColorClip(
            size=(SHORT_WIDTH, bar_height),
            color=style["bg_color"],
        )
        bg = bg.with_opacity(style["bg_opacity"])
        bg = bg.with_position(("center", pos_y))
        bg = bg.with_start(start).with_duration(duration)
        return bg
    except Exception:
        return None


# ── Hook Card ────────────────────────────────────────────────────────────────

def _add_hook_card(clip, hook_text: str):
    """Overlay hook text on the first seconds with gradient effect."""
    if not hook_text:
        return clip

    duration = min(HOOK_DURATION, clip.duration)

    # Semi-transparent overlay
    bg = ColorClip(
        size=(SHORT_WIDTH, SHORT_HEIGHT),
        color=(0, 0, 0),
    ).with_duration(duration).with_opacity(0.55)

    # Hook text using PIL rendering
    hook_img = _render_text_to_image(
        hook_text.upper(), HOOK_FONT_SIZE, "white", "black",
        6, SHORT_WIDTH - 100
    )
    txt = (
        ImageClip(hook_img)
        .with_position(("center", int(SHORT_HEIGHT * 0.38)))
        .with_duration(duration)
    )

    # Gold accent line
    accent = ColorClip(
        size=(250, 5),
        color=(255, 215, 0),
    ).with_duration(duration).with_position(
        ("center", int(SHORT_HEIGHT * 0.55))
    ).with_opacity(0.9)

    # "Watch till end" hint using PIL
    hint_img = _render_text_to_image(
        "▶ WATCH TILL END", 30, "#AAAAAA", "black",
        2, SHORT_WIDTH - 100
    )
    hint = (
        ImageClip(hint_img)
        .with_position(("center", int(SHORT_HEIGHT * 0.88)))
        .with_duration(duration)
    )

    hook_section = CompositeVideoClip([
        clip.with_duration(duration),
        bg, txt, accent, hint,
    ]).with_duration(duration)

    if clip.duration > duration:
        rest = clip.subclipped(duration, clip.duration)
        return concatenate_videoclips([hook_section, rest], method="compose")
    return hook_section


# ── Thumbnail Generator ─────────────────────────────────────────────────────

def _generate_thumbnail(video_path: str, segment, safe_title: str,
                        index: int) -> str:
    """Generate a thumbnail PNG from the best frame."""
    try:
        video = VideoFileClip(video_path)
        thumb_time = segment.start + (segment.end - segment.start) * 0.3
        thumb_time = min(thumb_time, video.duration - 0.1)

        frame = video.get_frame(thumb_time)
        video.close()

        img = Image.fromarray(frame)

        # Crop to 9:16
        target_ratio = SHORT_WIDTH / SHORT_HEIGHT
        src_w, src_h = img.size
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            new_w = int(src_h * target_ratio)
            x_offset = (src_w - new_w) // 2
            img = img.crop((x_offset, 0, x_offset + new_w, src_h))
        else:
            new_h = int(src_w / target_ratio)
            y_offset = (src_h - new_h) // 2
            img = img.crop((0, y_offset, src_w, y_offset + new_h))

        img = img.resize((SHORT_WIDTH, SHORT_HEIGHT), Image.LANCZOS)

        # Add dark gradient at bottom
        draw = ImageDraw.Draw(img, "RGBA")
        for y in range(SHORT_HEIGHT // 2, SHORT_HEIGHT):
            alpha = int(180 * (y - SHORT_HEIGHT // 2) / (SHORT_HEIGHT // 2))
            draw.rectangle([(0, y), (SHORT_WIDTH, y + 1)], fill=(0, 0, 0, alpha))

        # Add hook text on thumbnail
        hook_text = getattr(segment, 'hook_text', '') or getattr(segment, 'title', '')
        if hook_text:
            try:
                font = ImageFont.truetype(BOLD_FONT, 60)
            except Exception:
                font = ImageFont.load_default()

            words = hook_text.upper().split()
            lines = []
            current_line = ""
            for word in words:
                test = f"{current_line} {word}".strip()
                bbox = font.getbbox(test)
                if bbox[2] > SHORT_WIDTH - 100 and current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test
            if current_line:
                lines.append(current_line)

            text_draw = ImageDraw.Draw(img)
            y_pos = int(SHORT_HEIGHT * 0.72)
            for line in lines:
                bbox = font.getbbox(line)
                text_w = bbox[2] - bbox[0]
                x = (SHORT_WIDTH - text_w) // 2

                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        text_draw.text((x + dx, y_pos + dy), line, font=font, fill="black")
                text_draw.text((x, y_pos), line, font=font, fill="white")
                y_pos += bbox[3] - bbox[1] + 10

        thumb_path = str(THUMBNAILS_DIR / f"thumb_{index}_{safe_title}.png")
        img.save(thumb_path)
        return thumb_path

    except Exception as e:
        console.print(f"   [dim]⚠ Thumbnail failed: {e}[/dim]")
        return ""


# ── Utilities ────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'[\s]+', '_', safe)
    return safe[:50]


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
