"""
Local AI Segment Analyzer — uses Groq's free API (Llama 3) to identify
the most engaging segments. Free tier: 30 req/min, no credit card.

Sign up at https://console.groq.com (free, works globally).
"""

import json
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from src.config import SHORT_MAX_DURATION

console = Console()


@dataclass
class Segment:
    """An identified short-worthy segment."""
    start: float
    end: float
    title: str
    hook_text: str


SYSTEM_PROMPT = """You are a viral content strategist and video editor AI.

Your job is to analyze a video transcript and identify the most ENGAGING,
SHAREABLE, and SELF-CONTAINED segments that would make great YouTube Shorts.

Rules:
1. Each segment MUST be ≤ {max_duration} seconds long.
2. Each segment must be self-contained — it should make sense on its own.
3. Prioritize segments with: strong hooks, emotional moments, surprising facts,
   humor, actionable advice, or controversial takes.
4. The "hook_text" should be a punchy 1-liner that grabs attention in the first
   2 seconds (this will be overlaid on the video).
5. The "title" should be a catchy, clickable title for the short.

Return your answer as a JSON array with this exact structure:
[
  {{
    "start": <start_time_in_seconds>,
    "end": <end_time_in_seconds>,
    "title": "<catchy title for the short>",
    "hook_text": "<punchy hook text overlay>"
  }}
]

Return ONLY the JSON array, no other text or markdown formatting."""


def find_best_segments(
    transcript,
    num_shorts: int = 3,
    video_duration: float = 0.0,
    groq_api_key: str = "",
) -> list[Segment]:
    """
    Use Groq's free Llama 3 API to identify the best short-worthy segments.

    Args:
        transcript: The full video transcript with timestamps.
        num_shorts: Number of shorts to identify.
        video_duration: Total video duration for validation.
        groq_api_key: Groq API key (free at https://console.groq.com).

    Returns:
        List of Segment objects.
    """
    from groq import Groq

    console.print(f"\n[bold cyan]🧠 Analyzing transcript with Llama 3 (via Groq)...[/bold cyan]")
    console.print(f"   Looking for the top [bold]{num_shorts}[/bold] segments\n")

    # Build transcript text
    transcript_text = _format_transcript_for_llm(transcript)

    system = SYSTEM_PROMPT.format(max_duration=SHORT_MAX_DURATION)
    user_prompt = f"""Here is the video transcript with timestamps:

{transcript_text}

Total video duration: {video_duration:.1f} seconds.

Please identify the top {num_shorts} most engaging segments for YouTube Shorts.
Remember: each segment must be ≤ {SHORT_MAX_DURATION} seconds.
Return ONLY a JSON array."""

    # Call Groq (free Llama 3)
    console.print("   Calling Llama 3 via Groq (free)...")
    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    raw_content = response.choices[0].message.content.strip()

    segments = _parse_segments(raw_content, video_duration)
    _display_segments(segments)

    return segments


def _format_transcript_for_llm(transcript) -> str:
    """Format transcript segments with timestamps."""
    lines = []
    for seg in transcript.segments:
        timestamp = f"[{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        lines.append(f"{timestamp} {seg.text}")
    return "\n".join(lines)


def _parse_segments(raw_json: str, video_duration: float) -> list[Segment]:
    """Parse and validate the JSON response."""
    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_json = "\n".join(lines)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        console.print(f"[bold red]❌ Failed to parse LLM response as JSON: {e}[/bold red]")
        console.print(f"   Raw response:\n{raw_json[:500]}")
        return []

    segments = []
    for item in data:
        start = float(item["start"])
        end = float(item["end"])

        start = max(0, start)
        if video_duration > 0:
            end = min(end, video_duration)
        if (end - start) > SHORT_MAX_DURATION:
            end = start + SHORT_MAX_DURATION
        if (end - start) < 5:
            continue

        segments.append(Segment(
            start=start,
            end=end,
            title=item.get("title", "Untitled Short"),
            hook_text=item.get("hook_text", ""),
        ))

    return segments


def _display_segments(segments: list[Segment]):
    """Display identified segments in a nice table."""
    if not segments:
        console.print("[yellow]⚠ No segments found.[/yellow]")
        return

    table = Table(title="🎬 Identified Segments", show_lines=True)
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Time Range", style="white", width=18)
    table.add_column("Duration", style="green", width=8)
    table.add_column("Title", style="bold white", max_width=35)
    table.add_column("Hook", style="dim", max_width=40)

    for i, seg in enumerate(segments, 1):
        dur = seg.end - seg.start
        table.add_row(
            str(i),
            f"{_fmt_time(seg.start)} → {_fmt_time(seg.end)}",
            f"{dur:.0f}s",
            seg.title,
            seg.hook_text,
        )

    console.print()
    console.print(table)
    console.print()


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
