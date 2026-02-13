"""
YouTube Shorts AI Agent — CLI Entry Point

Usage:
    python main.py <YOUTUBE_URL> [--shorts N] [--dry-run]

Examples:
    python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --shorts 3
    python main.py "https://youtu.be/abc123" --shorts 2 --dry-run
"""

import argparse
import sys

from rich.console import Console

from src.config import MODE, OPENAI_API_KEY, GROQ_API_KEY
from src.pipeline import run

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="🎬 YouTube Shorts AI Agent — Turn any YouTube video into viral Shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python main.py "https://youtu.be/VIDEO_ID" --shorts 5
  python main.py "https://youtu.be/VIDEO_ID" --shorts 2 --dry-run
        """,
    )

    parser.add_argument(
        "url",
        help="YouTube video URL to process",
    )
    parser.add_argument(
        "--shorts", "-n",
        type=int,
        default=3,
        help="Number of shorts to generate (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything except the video editing step (for testing)",
    )

    args = parser.parse_args()

    # Validate API key based on mode
    if MODE == "local" and not GROQ_API_KEY:
        console.print(
            "[bold red]❌ GROQ_API_KEY not found![/bold red]\n"
            "   Get a free key at https://console.groq.com (no credit card).\n"
            "   Set it in your .env file or as an environment variable.",
        )
        sys.exit(1)
    elif MODE == "openai" and not OPENAI_API_KEY:
        console.print(
            "[bold red]❌ OPENAI_API_KEY not found![/bold red]\n"
            "   Set it in your .env file or as an environment variable.\n"
            "   See .env.example for the template.",
        )
        sys.exit(1)

    # Run the pipeline
    try:
        output_paths = run(
            url=args.url,
            num_shorts=args.shorts,
            dry_run=args.dry_run,
        )

        if output_paths:
            console.print(
                f"\n[bold green]🎉 Generated {len(output_paths)} short(s)![/bold green]"
            )
        elif not args.dry_run:
            console.print("\n[yellow]⚠ No shorts were generated.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        console.print_exception(show_locals=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
