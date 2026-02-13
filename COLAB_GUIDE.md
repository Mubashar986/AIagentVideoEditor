# 🚀 Google Colab Setup Guide

Run the entire YouTube Shorts AI Agent for **free** on Google Colab — no API costs, no VPS.

## What You Need

| Item | Where to Get It | Cost |
|------|----------------|------|
| Google Account | You already have one | Free |
| Groq API Key | [console.groq.com](https://console.groq.com) | Free, no credit card |

---

## Step-by-Step: Copy These Cells into Colab

### Open Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**
3. Go to **Runtime → Change runtime type → T4 GPU** ⚠️ (IMPORTANT!)

---

### Cell 1 — Clone the project & install dependencies

```python
# Always start from /content to avoid nested folders!
%cd /content
!rm -rf AIagentVideoEditor

# Clone repo
!git clone https://github.com/Mubashar986/AIagentVideoEditor.git
%cd /content/AIagentVideoEditor

# Install all dependencies
!pip install -q yt-dlp "moviepy>=2.0" faster-whisper groq gradio rich python-dotenv
```

> **Note:** If your repo isn't on GitHub, you can upload the project folder
> using Colab's file panel (📁 icon on the left) or mount Google Drive.

---

### Cell 2 — Set your API key & config

```python
import os

# Set your free Groq API key (get it at https://console.groq.com)
os.environ["GROQ_API_KEY"] = "gsk_YOUR_KEY_HERE"  # ← paste your key

# Use free local models
os.environ["MODE"] = "local"

# Whisper model size: tiny (fastest) | base (good) | small (better) | medium | large-v3 (best)
os.environ["WHISPER_LOCAL_MODEL"] = "base"

print("✅ Config set!")
```

---

### Cell 3 — Option A: Run via CLI (quick test)

```python
# Basic: generate 2 shorts
!python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --shorts 2

# With context for smarter picks (RECOMMENDED!):
!python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --shorts 3 --context "cricket match highlights - focus on best wickets and celebrations"

# More examples of --context:
# --context "motivational speech - find the most powerful quotes"
# --context "podcast interview - controversial takes and funny moments"
# --context "cooking tutorial - key tips and plating reveals"
# --context "gaming highlights - clutch plays and reactions"
```

---

### Cell 3 — Option B: Launch the Gradio Web UI (recommended!)

```python
# This launches a web UI and gives you a PUBLIC shareable link
!python app.py
```

After running, you'll see:
```
Running on public URL: https://xxxxx.gradio.live
```
**Click that link** — it opens a web app where you can:
- Paste any YouTube URL
- Pick number of shorts
- **Add video context** for smarter segment picks
- Download the generated shorts with animated captions

---

### Cell 4 — Download your shorts (if using CLI)

```python
# List generated shorts
import os
from google.colab import files as colab_files

output_dir = "output"
for f in os.listdir(output_dir):
    if f.endswith(".mp4"):
        print(f"📁 {f}")
        colab_files.download(os.path.join(output_dir, f))
```

---

## Alternative: Upload Project Without GitHub

If you don't want to push to GitHub:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive
!cp -r "/content/drive/MyDrive/AIagent" /content/AIagent
%cd /content/AIagent
```

Or just drag-and-drop the project folder into Colab's file panel.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No GPU available" | Go to Runtime → Change runtime type → T4 GPU |
| Groq rate limit | Wait 60 seconds and try again (free tier = 30 req/min) |
| Video download fails | Try a different URL, or update yt-dlp: `!pip install -U yt-dlp` |
| Out of disk space | Delete old files: `!rm -rf downloads/* output/*` |
| Session disconnects | Normal for free Colab — re-run from Cell 1 |
| Nested folders | Always start Cell 1 with `%cd /content` |
