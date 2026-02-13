"""
YouTube Shorts AI Agent — Gradio Web UI.

Run with: python app.py
Opens a shareable web interface for generating shorts.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import OUTPUT_DIR


def process_video(url: str, num_shorts: int, groq_key: str, video_context: str):
    """Process a video and return generated shorts."""
    if not url or not url.strip():
        return None, "❌ Please enter a YouTube URL"

    if not groq_key or not groq_key.strip():
        return None, "❌ Please enter your Groq API key (free at console.groq.com)"

    # Set environment variables for this run
    os.environ["GROQ_API_KEY"] = groq_key.strip()
    os.environ["MODE"] = "local"

    # Re-import to pick up the new key
    from src.pipeline import run

    try:
        output_paths = run(
            url=url.strip(),
            num_shorts=int(num_shorts),
            video_context=video_context.strip() if video_context else "",
        )

        if output_paths:
            status = f"✅ Generated {len(output_paths)} short(s)!"
            return output_paths, status
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

            **How it works:** Paste a URL → AI picks the best moments → Get 
            upload-ready vertical shorts with animated captions.

            **New:** Add a video context to get smarter picks 
            (e.g. "cricket match — focus on wickets and celebrations").
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    max_lines=1,
                )
            with gr.Column(scale=1):
                num_shorts = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Number of Shorts",
                )

        with gr.Row():
            with gr.Column(scale=2):
                groq_key = gr.Textbox(
                    label="Groq API Key (free)",
                    placeholder="gsk_... (get free key at console.groq.com)",
                    type="password",
                    max_lines=1,
                )
            with gr.Column(scale=2):
                video_context = gr.Textbox(
                    label="Video Context (optional)",
                    placeholder="e.g. 'cricket highlights - best wickets' or 'motivational speech - powerful quotes'",
                    max_lines=1,
                )

        generate_btn = gr.Button(
            "🚀 Generate Shorts",
            variant="primary",
            size="lg",
        )

        status_text = gr.Textbox(
            label="Status",
            interactive=False,
        )

        output_videos = gr.Files(
            label="Generated Shorts",
        )

        generate_btn.click(
            fn=process_video,
            inputs=[url_input, num_shorts, groq_key, video_context],
            outputs=[output_videos, status_text],
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Get a free Groq API key at [console.groq.com](https://console.groq.com)
            - Use video context for better results (e.g. "cooking tutorial" or "podcast interview")
            - Longer videos (10+ min) produce better variety in shorts
            - Each short is 25-55 seconds with animated captions
            """
        )

    app.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    create_ui()
