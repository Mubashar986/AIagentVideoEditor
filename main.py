"""
YouTube Shorts AI Agent — CLI entry point.

Usage:
    python main.py "https://youtube.com/watch?v=..." --shorts 3
    python main.py "https://youtube.com/watch?v=..." --shorts 2 --context "cricket match, focus on best wickets"
    python main.py "https://youtube.com/watch?v=..." --dry-run
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run
from src.config import MODE, OPENAI_API_KEY, GROQ_API_KEY

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="🎬 YouTube Shorts AI Agent — auto-generate viral shorts from any video.",
    )
    parser.add_argument(
        "url",
        help="YouTube video URL",
    )
    parser.add_argument(
        "--shorts",
        type=int,
        default=3,
        help="Number of shorts to generate (default: 3)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Describe the video content for better segment picks "
             "(e.g. 'cricket match highlights' or 'motivational speech')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify segments without rendering (preview mode)",
    )

    args = parser.parse_args()

    # Validate API key based on mode
    if MODE == "local" and not GROQ_API_KEY:
        console.print(
            "❌ GROQ_API_KEY not found!\n"
            "   Get a free key at https://console.groq.com\n"
            "   Then set it: export GROQ_API_KEY='gsk_...'"
        )
        sys.exit(1)
    elif MODE == "openai" and not OPENAI_API_KEY:
        console.print(
            "❌ OPENAI_API_KEY not found!\n"
            "   Set it: export OPENAI_API_KEY='sk-...'"
        )
        sys.exit(1)

    # Run the pipeline
    try:
        output_paths = run(
            url=args.url,
            num_shorts=args.shorts,
            dry_run=args.dry_run,
            video_context=args.context,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)

    if output_paths:
        console.print(f"\n[bold green]🎉 All done! {len(output_paths)} short(s) ready.[/bold green]")
    else:
        console.print("\n[dim]No videos generated.[/dim]")


if __name__ == "__main__":
    main()
