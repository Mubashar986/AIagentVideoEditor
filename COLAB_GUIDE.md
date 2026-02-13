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

### Cell 1 — Clone & install

```python
%cd /content
!rm -rf AIagentVideoEditor

!git clone https://github.com/Mubashar986/AIagentVideoEditor.git
%cd /content/AIagentVideoEditor

!pip install -q yt-dlp "moviepy>=2.0" faster-whisper groq gradio rich python-dotenv numpy Pillow
```

---

### Cell 2 — Set API key

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_YOUR_KEY_HERE"  # ← paste your key
os.environ["MODE"] = "local"
os.environ["WHISPER_LOCAL_MODEL"] = "base"
print("✅ Config set!")
```

---

### Cell 3 — Option A: CLI (quick test)

```python
# Basic:
!python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --shorts 2

# With context + style:
!python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --shorts 3 \
    --context "cricket match - best wickets and celebrations" \
    --style beast

# Batch mode (multiple videos):
!python main.py "URL1" "URL2" --shorts 2 --batch \
    --context "highlights" --style hormozi
```

#### Caption Styles:
| Style | Look |
|-------|------|
| `hormozi` | ALL CAPS, gold highlight, 3 words/group |
| `beast` | Bold, red highlight, 2 words/group |
| `subtle` | Small white text at bottom |
| `karaoke` | Medium text, white on dim |

---

### Cell 3 — Option B: Gradio Web UI (recommended!)

```python
!python app.py
```

Click the public URL → paste URL, pick style, add context, generate!

---

### Cell 4 — Download shorts + thumbnails

```python
import os
from google.colab import files as colab_files

for folder in ["output", "thumbnails"]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            print(f"📁 {path}")
            colab_files.download(path)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No GPU available" | Runtime → Change runtime type → T4 GPU |
| Groq rate limit | Wait 60s (free = 30 req/min) |
| Video download fails | `!pip install -U yt-dlp` |
| Out of disk space | `!rm -rf downloads/* output/* thumbnails/*` |
| Session disconnects | Normal for free Colab — re-run from Cell 1 |
| Nested folders | Always start with `%cd /content` |
