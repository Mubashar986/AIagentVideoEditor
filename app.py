"""
YouTube Shorts AI Agent — Gradio Web Frontend
Run this to get a web UI with a shareable link.

Usage:
    python app.py
"""

import os
import gradio as gr
from pathlib import Path

# Ensure we're in local/free mode
os.environ.setdefault("MODE", "local")

from src.config import OUTPUT_DIR, GROQ_API_KEY
from src.pipeline import run


def process_video(url: str, num_shorts: int, groq_key: str, progress=gr.Progress()):
    """Main processing function called by Gradio."""

    # Set the API key if provided
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    if not url:
        return [], "❌ Please enter a YouTube URL."

    if not groq_key and not GROQ_API_KEY:
        return [], "❌ Please enter your Groq API key (free at console.groq.com)."

    try:
        progress(0.1, desc="📥 Downloading video...")
        # Run the full pipeline
        output_paths = run(
            url=url,
            num_shorts=int(num_shorts),
            dry_run=False,
        )

        if not output_paths:
            return [], "⚠️ No shorts were generated. Try a different video."

        # Return video files for display
        status = f"✅ Generated {len(output_paths)} short(s)! Download them below."
        return output_paths, status

    except Exception as e:
        return [], f"❌ Error: {str(e)}"


def create_ui():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="🎬 YouTube Shorts AI Agent",
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="blue",
        ),
        css="""
        .main-header { text-align: center; margin-bottom: 1rem; }
        .output-videos { min-height: 200px; }
        """,
    ) as app:

        # Header
        gr.Markdown(
            """
            # 🎬 YouTube Shorts AI Agent
            ### Turn any YouTube video into viral Shorts — 100% free
            **Powered by:** faster-whisper (transcription) + Llama 3 via Groq (AI analysis) + MoviePy (editing)
            """,
            elem_classes="main-header",
        )

        with gr.Row():
            with gr.Column(scale=2):
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    max_lines=1,
                )

            with gr.Column(scale=1):
                num_shorts = gr.Slider(
                    label="Number of Shorts",
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                )

        with gr.Accordion("🔑 API Key (Groq — free)", open=True):
            gr.Markdown(
                "Get a **free** API key at [console.groq.com](https://console.groq.com) "
                "(no credit card needed)."
            )
            groq_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password",
                max_lines=1,
            )

        generate_btn = gr.Button(
            "🚀 Generate Shorts",
            variant="primary",
            size="lg",
        )

        # Output area
        status_text = gr.Markdown("", label="Status")

        with gr.Row():
            output_videos = gr.Files(
                label="📁 Generated Shorts (click to download)",
                file_count="multiple",
                elem_classes="output-videos",
            )

        # How it works
        with gr.Accordion("ℹ️ How It Works", open=False):
            gr.Markdown(
                """
                1. **Download** — Video is downloaded using yt-dlp
                2. **Transcribe** — Audio is transcribed with faster-whisper (GPU)
                3. **AI Analysis** — Llama 3 identifies the most viral-worthy segments
                4. **Edit** — Each segment is cropped to 9:16, captions are burned in
                5. **Export** — Upload-ready .mp4 shorts in the output folder
                """
            )

        # Wire up the button
        generate_btn.click(
            fn=process_video,
            inputs=[url_input, num_shorts, groq_key],
            outputs=[output_videos, status_text],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        share=True,  # Creates a public shareable link
        server_name="0.0.0.0",
        server_port=7860,
    )
