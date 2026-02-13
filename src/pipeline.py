"""
Pipeline Orchestrator — chains Download → Transcribe → Analyze → Edit.
Automatically picks free (local) or paid (OpenAI) models based on config.
"""

from rich.console import Console
from rich.panel import Panel

from src.config import MODE, GROQ_API_KEY, OPENAI_API_KEY, WHISPER_LOCAL_MODEL
from src.downloader import download_video
from src.editor import create_short

console = Console()


def run(url: str, num_shorts: int = 3, dry_run: bool = False) -> list[str]:
    """
    Execute the full YouTube → Shorts pipeline.

    Args:
        url: YouTube video URL.
        num_shorts: Number of shorts to generate.
        dry_run: If True, skip the video editing step.

    Returns:
        List of output file paths.
    """
    mode_label = "🆓 FREE (local models)" if MODE == "local" else "💳 OpenAI (paid)"

    console.print(Panel.fit(
        "[bold white]🚀 YouTube Shorts AI Agent[/bold white]\n"
        f"   URL:    {url}\n"
        f"   Shorts: {num_shorts}\n"
        f"   Mode:   {mode_label}\n"
        f"   Run:    {'🧪 DRY RUN' if dry_run else '🎬 FULL RUN'}",
        border_style="bright_cyan",
    ))

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
        )
    else:
        from src.analyzer import find_best_segments
        segments = find_best_segments(
            transcript=transcript,
            num_shorts=num_shorts,
            video_duration=result.duration,
        )

    if not segments:
        console.print("[bold red]❌ No viable segments found. Aborting.[/bold red]")
        return []

    # ── Stage 4: Video Editing ─────────────────────────────────────────
    if dry_run:
        console.print(
            "\n[bold yellow]🧪 DRY RUN — Skipping video editing.[/bold yellow]"
        )
        console.print("   The segments above would be turned into shorts.\n")
        return []

    output_paths = []
    for i, segment in enumerate(segments, 1):
        path = create_short(
            video_path=result.video_path,
            segment=segment,
            transcript=transcript,
            index=i,
        )
        output_paths.append(path)

    _print_summary(output_paths)
    return output_paths


def _print_summary(paths: list[str]):
    """Print a final summary."""
    console.print(Panel.fit(
        "[bold green]✅ All shorts generated![/bold green]\n\n"
        + "\n".join(f"   📁 {p}" for p in paths)
        + "\n\n   Ready to upload to YouTube Shorts!",
        title="🎉 Complete",
        border_style="green",
    ))
