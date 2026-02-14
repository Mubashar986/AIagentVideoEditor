"""
Local AI Segment Analyzer — uses Groq's free API (Llama 3) to identify
the most engaging segments. Free tier: 30 req/min, no credit card.

Now supports video_context for content-aware picks (e.g. "cricket highlights").
Auto-truncates long transcripts to stay within Groq's token limits.
"""

import json
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from src.config import SHORT_MIN_DURATION, SHORT_MAX_DURATION, GROQ_MODEL

console = Console()

# Groq free tier limit: ~6000 tokens/request
# System prompt + user prompt overhead ≈ 800 tokens
# Leave room for response ≈ 1500 tokens
MAX_TRANSCRIPT_CHARS = 7000  # ~2000 tokens for transcript


@dataclass
class Segment:
    """An identified short-worthy segment."""
    start: float
    end: float
    title: str
    hook_text: str


SYSTEM_PROMPT = """You are a viral content editor AI for YouTube Shorts.

You analyze video transcripts and pick the BEST, most VIRAL segments.

STRICT RULES:
1. Each segment MUST be between {min_duration}–{max_duration} seconds long.
   Prefer segments closer to {max_duration} seconds — longer = more engagement.
2. Segments MUST NOT OVERLAP — minimum 10 second gap between segments.
3. SPREAD segments across the ENTIRE video — do NOT cluster them at the start.
4. Each segment must be SELF-CONTAINED — it should make complete sense alone.
5. The "hook_text" is a PUNCHY 1-liner shown in the first 1.5 seconds.
   Make it provocative, emotional, or surprising — this decides if people watch.
6. The "title" should be a clickable, curiosity-driven title.

WHAT MAKES A VIRAL SHORT:
- Strong emotional hook in the first 2 seconds
- Surprising facts, controversial takes, or "wait what?" moments
- Complete story arc: setup → tension → payoff
- Actionable advice or life-changing insight
- Humor, drama, or raw authenticity

{video_context}

Return ONLY a JSON array with this exact structure:
[
  {{
    "start": <start_seconds>,
    "end": <end_seconds>,
    "title": "<catchy clickable title>",
    "hook_text": "<punchy hook overlay text>"
  }}
]

Return ONLY the JSON array, nothing else."""


def find_best_segments(
    transcript,
    num_shorts: int = 3,
    video_duration: float = 0.0,
    groq_api_key: str = "",
    video_context: str = "",
) -> list[Segment]:
    """
    Use Groq's free Llama 3 API to find the best short-worthy segments.

    Auto-truncates long transcripts to stay within Groq's free tier limits.
    """
    from groq import Groq

    console.print(f"\n[bold cyan]🧠 Analyzing transcript with Llama 3 (via Groq)...[/bold cyan]")
    console.print(f"   Looking for the top [bold]{num_shorts}[/bold] segments")
    if video_context:
        console.print(f"   Context: [italic]{video_context}[/italic]")
    console.print()

    # Build context section for the prompt
    context_section = ""
    if video_context:
        context_section = f"""
IMPORTANT VIDEO CONTEXT FROM THE USER:
\"{video_context}\"
Use this context to decide WHAT to look for. Adapt your segment selection to match the user's intent.
"""

    transcript_text = _format_transcript_for_llm(transcript)

    # Auto-truncate if transcript is too long
    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        original_len = len(transcript_text)
        transcript_text = _smart_truncate(transcript_text, MAX_TRANSCRIPT_CHARS, video_duration)
        console.print(
            f"   [yellow]⚠ Transcript too long ({original_len:,} chars) — "
            f"truncated to {len(transcript_text):,} chars to fit Groq limits[/yellow]"
        )

    system = SYSTEM_PROMPT.format(
        min_duration=SHORT_MIN_DURATION,
        max_duration=SHORT_MAX_DURATION,
        video_context=context_section,
    )
    user_prompt = f"""Video transcript with timestamps:

{transcript_text}

Total video duration: {video_duration:.1f} seconds.

Find the top {num_shorts} BEST, non-overlapping segments for YouTube Shorts.
Each must be {SHORT_MIN_DURATION}–{SHORT_MAX_DURATION} seconds, spread across the whole video.
Return ONLY a JSON array."""

    console.print("   Calling Llama 3 via Groq (free)...")
    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    raw_content = response.choices[0].message.content.strip()

    segments = _parse_segments(raw_content, video_duration)
    segments = _remove_overlapping(segments)
    _display_segments(segments)

    return segments


def _format_transcript_for_llm(transcript) -> str:
    """Format transcript segments with timestamps."""
    lines = []
    for seg in transcript.segments:
        timestamp = f"[{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        lines.append(f"{timestamp} {seg.text}")
    return "\n".join(lines)


def _smart_truncate(transcript_text: str, max_chars: int, video_duration: float) -> str:
    """
    Intelligently truncate a long transcript while preserving coverage
    of the entire video. Instead of cutting off the end, we sample
    evenly from beginning, middle, and end.
    """
    lines = transcript_text.split("\n")
    total_lines = len(lines)

    if total_lines <= 10:
        return transcript_text[:max_chars]

    # Calculate how many lines we can keep
    avg_line_len = len(transcript_text) / total_lines
    target_lines = int(max_chars / avg_line_len)
    target_lines = max(target_lines, 10)

    if target_lines >= total_lines:
        return transcript_text

    # Strategy: keep lines evenly spread across the transcript
    # This ensures the AI sees content from ALL parts of the video
    step = total_lines / target_lines
    selected_indices = []
    pos = 0.0
    while pos < total_lines and len(selected_indices) < target_lines:
        idx = int(pos)
        if idx < total_lines:
            selected_indices.append(idx)
        pos += step

    # Always include first and last few lines for context
    must_include = set(range(min(3, total_lines)))  # first 3
    must_include.update(range(max(0, total_lines - 3), total_lines))  # last 3
    all_indices = sorted(set(selected_indices) | must_include)

    selected_lines = [lines[i] for i in all_indices if i < total_lines]
    result = "\n".join(selected_lines)

    # Final safety check
    if len(result) > max_chars:
        result = result[:max_chars]

    return result


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
        if (end - start) < SHORT_MIN_DURATION:
            # Try extending to min duration if there's room
            end = start + SHORT_MIN_DURATION
            if video_duration > 0 and end > video_duration:
                continue  # can't extend, skip
        if (end - start) < 10:
            continue

        segments.append(Segment(
            start=start,
            end=end,
            title=item.get("title", "Untitled Short"),
            hook_text=item.get("hook_text", ""),
        ))

    return segments


def _remove_overlapping(segments: list[Segment]) -> list[Segment]:
    """Remove overlapping segments, keeping the first one in each conflict."""
    if not segments:
        return segments

    segments.sort(key=lambda s: s.start)
    result = [segments[0]]

    for seg in segments[1:]:
        last = result[-1]
        if seg.start >= last.end + 10:
            result.append(seg)
        else:
            console.print(
                f"   [dim]Skipped overlapping segment "
                f"{_fmt_time(seg.start)}→{_fmt_time(seg.end)}[/dim]"
            )

    return result


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
