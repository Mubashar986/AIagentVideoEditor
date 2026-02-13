"""
YouTube Shorts AI Agent — CLI entry point.

Usage:
    python main.py "URL" --shorts 3
    python main.py "URL" --shorts 2 --context "cricket highlights"
    python main.py "URL" --shorts 3 --style beast
    python main.py "URL1" "URL2" --shorts 2 --batch
    python main.py "URL" --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run, run_batch
from src.config import MODE, OPENAI_API_KEY, GROQ_API_KEY

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="🎬 YouTube Shorts AI Agent — generate viral shorts from any video.",
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="YouTube video URL(s). Pass multiple URLs with --batch.",
    )
    parser.add_argument(
        "--shorts",
        type=int,
        default=3,
        help="Number of shorts per video (default: 3)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Describe video content for smarter picks "
             "(e.g. 'cricket match - best wickets')",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="",
        choices=["hormozi", "beast", "subtle", "karaoke", ""],
        help="Caption style preset (default: hormozi)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple URLs in batch mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify segments without rendering (preview mode)",
    )

    args = parser.parse_args()

    # Validate API key
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

    try:
        if args.batch or len(args.urls) > 1:
            # Batch mode: multiple URLs
            output_paths = run_batch(
                urls=args.urls,
                num_shorts=args.shorts,
                video_context=args.context,
                caption_style=args.style,
            )
        else:
            # Single URL
            output_paths = run(
                url=args.urls[0],
                num_shorts=args.shorts,
                dry_run=args.dry_run,
                video_context=args.context,
                caption_style=args.style,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if output_paths:
        console.print(f"\n[bold green]🎉 All done! {len(output_paths)} short(s) ready.[/bold green]")
    else:
        console.print("\n[dim]No videos generated.[/dim]")


if __name__ == "__main__":
    main()
