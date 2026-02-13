"""
YouTube Shorts AI Agent — Gradio Web UI.

Features:
- Video context for smart segment picks
- Caption style presets
- Background music from URL
- Thumbnail generation
- Batch URL processing

Run with: python app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import OUTPUT_DIR, THUMBNAILS_DIR


def process_video(url: str, num_shorts: int, groq_key: str,
                   video_context: str, caption_style: str, music_url: str):
    """Process a video and return generated shorts."""
    if not url or not url.strip():
        return None, "❌ Please enter a YouTube URL"

    if not groq_key or not groq_key.strip():
        return None, "❌ Please enter your Groq API key (free at console.groq.com)"

    os.environ["GROQ_API_KEY"] = groq_key.strip()
    os.environ["MODE"] = "local"

    from src.pipeline import run, run_batch

    try:
        urls = [u.strip() for u in url.strip().split("\n") if u.strip()]

        all_paths = []
        if len(urls) > 1:
            all_paths = run_batch(
                urls=urls,
                num_shorts=int(num_shorts),
                video_context=video_context.strip() if video_context else "",
                caption_style=caption_style.lower() if caption_style else "",
                music_url=music_url.strip() if music_url else "",
            )
        else:
            all_paths = run(
                url=urls[0],
                num_shorts=int(num_shorts),
                video_context=video_context.strip() if video_context else "",
                caption_style=caption_style.lower() if caption_style else "",
                music_url=music_url.strip() if music_url else "",
            )

        # Collect thumbnails
        thumb_paths = []
        if THUMBNAILS_DIR.exists():
            thumb_paths = [
                str(THUMBNAILS_DIR / f) for f in os.listdir(THUMBNAILS_DIR)
                if f.endswith(".png")
            ]

        if all_paths:
            status = f"✅ Generated {len(all_paths)} short(s)!"
            if thumb_paths:
                status += f" + {len(thumb_paths)} thumbnail(s)"
            return all_paths + thumb_paths, status
        else:
            return None, "⚠ No segments found — try a different video or context."

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_ui():
    """Build and launch the Gradio interface."""

    with gr.Blocks(
        title="🎬 YouTube Shorts AI Agent",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="amber",
        ),
    ) as app:

        gr.Markdown(
            """
            # 🎬 YouTube Shorts AI Agent
            ### Transform any video into viral-ready shorts — 100% free

            **Features:** Word-by-word captions • 4 caption styles • Ken Burns zoom • 
            Background music • Auto thumbnails • Batch processing
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    label="YouTube URL(s)",
                    placeholder="https://www.youtube.com/watch?v=...\n(one per line for batch)",
                    lines=2,
                )
            with gr.Column(scale=1):
                num_shorts = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Shorts per Video",
                )

        with gr.Row():
            groq_key = gr.Textbox(
                label="Groq API Key (free)",
                placeholder="gsk_...",
                type="password",
                max_lines=1,
                scale=2,
            )
            video_context = gr.Textbox(
                label="Video Context",
                placeholder="e.g. 'cricket - best wickets'",
                max_lines=1,
                scale=2,
            )

        with gr.Row():
            caption_style = gr.Dropdown(
                label="Caption Style",
                choices=["hormozi", "beast", "subtle", "karaoke"],
                value="hormozi",
                scale=1,
            )
            music_url = gr.Textbox(
                label="🎵 Background Music URL (optional)",
                placeholder="YouTube/SoundCloud URL or direct MP3 link",
                max_lines=1,
                scale=3,
            )

        generate_btn = gr.Button(
            "🚀 Generate Shorts",
            variant="primary",
            size="lg",
        )

        status_text = gr.Textbox(label="Status", interactive=False)
        output_files = gr.Files(label="Generated Files (Shorts + Thumbnails)")

        generate_btn.click(
            fn=process_video,
            inputs=[url_input, num_shorts, groq_key, video_context, caption_style, music_url],
            outputs=[output_files, status_text],
        )

        gr.Markdown(
            """
            ---
            | Style | Look | Music Tips |
            |-------|------|------------|
            | **hormozi** | Gold highlight, ALL CAPS, 3 words | Lo-fi beats for motivational |
            | **beast** | Red highlight, bold, 2 words | Energetic EDM for sports/gaming |
            | **subtle** | Small white at bottom | Calm ambient for docs |
            | **karaoke** | White on dim, 4 words | Any genre |

            🎵 **Music:** Paste any YouTube music URL, SoundCloud link, or direct .mp3 URL.
            The music plays at 15% volume under the original video audio.
            """
        )

    app.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    create_ui()
