"""
Pipeline — full end-to-end: download → transcribe → AI analysis → edit.
Supports both 'local' (free) and 'openai' (paid) modes.
"""

from rich.console import Console
from rich.panel import Panel

from src.config import (
    MODE,
    GROQ_API_KEY,
    WHISPER_LOCAL_MODEL,
)
from src.downloader import download_video
from src.editor import create_short

console = Console()


def run(
    url: str,
    num_shorts: int = 3,
    dry_run: bool = False,
    video_context: str = "",
) -> list[str]:
    """
    Full pipeline: URL → finished YouTube Shorts.

    Args:
        url: YouTube video URL.
        num_shorts: Number of shorts to generate.
        dry_run: If True, identify segments without rendering.
        video_context: User description of the video content for
                       smarter segment selection (e.g. "cricket highlights").

    Returns:
        List of paths to exported .mp4 short files.
    """
    mode_label = "🆓 FREE (local models)" if MODE == "local" else "💳 OpenAI (paid)"

    info = (
        f"🚀 YouTube Shorts AI Agent\n"
        f"   URL:    {url}\n"
        f"   Shorts: {num_shorts}\n"
        f"   Mode:   {mode_label}\n"
        f"   Run:    {'🧪 DRY RUN (no render)' if dry_run else '🎬 FULL RUN'}"
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
        )
        output_paths.append(path)

    console.print(
        f"\n[bold green]🎉 Done! Generated {len(output_paths)} short(s).[/bold green]"
    )
    for p in output_paths:
        console.print(f"   📁 {p}")

    return output_paths
