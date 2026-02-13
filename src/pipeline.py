"""
Pipeline — full end-to-end: download → transcribe → AI analysis → edit.
Supports both 'local' (free) and 'openai' (paid) modes.
Supports single URL and batch playlist processing.
"""

from rich.console import Console
from rich.panel import Panel

from src.config import (
    MODE,
    GROQ_API_KEY,
    WHISPER_LOCAL_MODEL,
    CAPTION_STYLE,
)
from src.downloader import download_video
from src.editor import create_short

console = Console()


def run(
    url: str,
    num_shorts: int = 3,
    dry_run: bool = False,
    video_context: str = "",
    caption_style: str = "",
) -> list[str]:
    """
    Full pipeline: URL → finished YouTube Shorts.

    Args:
        url: YouTube video URL.
        num_shorts: Number of shorts to generate.
        dry_run: If True, identify segments without rendering.
        video_context: Description of video content for smarter picks.
        caption_style: Caption preset (hormozi/beast/subtle/karaoke).

    Returns:
        List of paths to exported .mp4 short files.
    """
    mode_label = "🆓 FREE (local models)" if MODE == "local" else "💳 OpenAI (paid)"
    style_label = caption_style or CAPTION_STYLE

    info = (
        f"🚀 YouTube Shorts AI Agent\n"
        f"   URL:    {url}\n"
        f"   Shorts: {num_shorts}\n"
        f"   Mode:   {mode_label}\n"
        f"   Style:  {style_label}\n"
        f"   Run:    {'🧪 DRY RUN' if dry_run else '🎬 FULL RUN'}"
    )
    if video_context:
        info += f"\n   Context: {video_context}"

    console.print(Panel(info))

    # ── Stage 1: Download ──────────────────────────────────────────────
    result = download_video(url)

    # ── Stage 2: Transcribe ────────────────────────────────────────────
    if MODE == "local":
        from src.transcriber_local import transcribe
        transcript = transcribe(result.video_path, model_size=WHISPER_LOCAL_MODEL)
    else:
        from src.transcriber import transcribe
        transcript = transcribe(result.video_path)

    # ── Stage 3: AI Analysis ───────────────────────────────────────────
    if MODE == "local":
        from src.analyzer_local import find_best_segments
        segments = find_best_segments(
            transcript=transcript,
            num_shorts=num_shorts,
            video_duration=result.duration,
            groq_api_key=GROQ_API_KEY,
            video_context=video_context,
        )
    else:
        from src.analyzer import find_best_segments
        segments = find_best_segments(
            transcript=transcript,
            num_shorts=num_shorts,
            video_duration=result.duration,
        )

    if dry_run:
        console.print("\n[bold yellow]🧪 Dry run — skipping video export.[/bold yellow]")
        return []

    # ── Stage 4: Edit & Export ─────────────────────────────────────────
    output_paths = []
    for i, segment in enumerate(segments, 1):
        path = create_short(
            video_path=result.video_path,
            segment=segment,
            transcript=transcript,
            index=i,
            caption_style=caption_style,
        )
        output_paths.append(path)

    console.print(
        f"\n[bold green]🎉 Done! Generated {len(output_paths)} short(s).[/bold green]"
    )
    for p in output_paths:
        console.print(f"   📁 {p}")

    return output_paths


def run_batch(
    urls: list[str],
    num_shorts: int = 2,
    video_context: str = "",
    caption_style: str = "",
) -> list[str]:
    """
    Process multiple video URLs in batch.

    Args:
        urls: List of YouTube URLs.
        num_shorts: Shorts per video.
        video_context: Context description.
        caption_style: Caption preset.

    Returns:
        All output paths across all videos.
    """
    console.print(
        f"\n[bold cyan]📦 Batch Mode: Processing {len(urls)} videos[/bold cyan]\n"
    )

    all_paths = []
    for i, url in enumerate(urls, 1):
        console.print(f"\n[bold white]── Video {i}/{len(urls)} ──[/bold white]")
        try:
            paths = run(
                url=url.strip(),
                num_shorts=num_shorts,
                video_context=video_context,
                caption_style=caption_style,
            )
            all_paths.extend(paths)
        except Exception as e:
            console.print(f"[red]❌ Failed: {e}[/red]")
            continue

    console.print(
        f"\n[bold green]🎉 Batch complete! "
        f"{len(all_paths)} total short(s) generated.[/bold green]"
    )
    return all_paths
