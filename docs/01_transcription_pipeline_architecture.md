# Audio/Video Transcription Pipeline Architecture

> A senior-engineer deep dive — from first principles to production patterns.  
> Based on YOUR actual codebase: `AIagent/src/`

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [Mental Model](#2-mental-model)
3. [Internals — How It Works](#3-internals--how-it-works)
4. [Industry Usage](#4-industry-usage)
5. [Tradeoffs](#5-tradeoffs)
6. [Building Perspective](#6-building-perspective)
7. [Common Mistakes](#7-common-mistakes)
8. [Engineering Best Practices](#8-engineering-best-practices)
9. [Learning Path](#9-learning-path)
10. [Real Example — Your Codebase Dissected](#10-real-example--your-codebase-dissected)

---

## 1. Big Picture

### What Problem Does This Solve?

Video is **opaque data**. You can't search it, filter it, or reason about it programmatically. A 30-minute YouTube video is just a blob of compressed pixels and waveforms. To build *any* intelligent system on top of video — search, summarization, clip extraction, content moderation — you need to **convert unstructured media into structured text**.

Transcription is the **bridge** between raw media and actionable data.

### Why Does This Exist?

In your system specifically, you need transcription because:

1. **Your AI analyzer (GPT/Llama3) can't watch video.** LLMs operate on text. The only way to let an LLM pick "the most viral 55s segment" is to give it a timestamped text transcript.
2. **Captions require word-level timing.** Your editor overlays animated word-by-word captions. That requires knowing *exactly* when each word was spoken — down to the millisecond.
3. **Context window efficiency.** A 30-min video has ~5,000 words of speech. That's ~6K tokens — fits in a single LLM context window. The equivalent raw audio would be ~300MB of float32 samples. Text is the compression.

### Where Does It Fit in a Real Production System?

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐
│  downloader  │───►│   transcriber    │───►│    analyzer       │───►│   editor   │
│  (yt-dlp)    │    │ (Whisper/API)    │    │ (GPT/Llama3)     │    │ (moviepy)  │
│              │    │                  │    │                  │    │            │
│  URL → .mp4  │    │ .mp4 → Transcript│    │ Transcript →     │    │ Segment +  │
│              │    │ (text+timestamps)│    │ Segments[]       │    │ Transcript │
│              │    │                  │    │                  │    │ → .mp4     │
└──────────────┘    └──────────────────┘    └──────────────────┘    └────────────┘
```

The transcriber is **Stage 2** of your 4-stage pipeline. It's the data transformation layer that makes everything downstream possible.

---

## 2. Mental Model

### The Intuitive Analogy

Think of transcription like a **court stenographer**.

A courtroom has raw audio (people talking). The stenographer converts that into a timestamped written record. Lawyers (your analyzer LLM) then read the record to find key moments. They can't listen to 8 hours of audio — but they *can* scan a transcript.

Your transcription pipeline is an **automated stenographer** with two modes:
- **Cloud mode (OpenAI)**: You hire a professional transcription service. High quality, costs money, you don't control the infrastructure.
- **Local mode (faster-whisper)**: You buy your own stenography machine and operate it yourself. Free per-use, but you need the hardware (GPU).

### How Experienced Engineers Think About It

Senior engineers think about transcription in terms of **four dimensions**:

| Dimension | Question | Your System |
|-----------|----------|-------------|
| **Latency** | How fast do I need the result? | Batch (offline) — minutes are fine |
| **Accuracy** | What word error rate is acceptable? | Moderate — it's for segment *selection*, not medical records |
| **Cost** | Per-minute of audio, what does it cost? | Local = $0 (GPU time), API = ~$0.006/min |
| **Scale** | How many hours/day of audio? | Low — single videos at a time |

Your system operates in **batch/offline mode**. This is the simplest and most forgiving quadrant. You don't need real-time streaming, you don't need sub-1% WER, and you're processing one video at a time. This informed every design choice in your code.

---

## 3. Internals — How It Works

### The Whisper Model — What's Actually Happening Inside

Whisper (OpenAI's speech recognition model) is a **sequence-to-sequence transformer** trained on 680,000 hours of multilingual audio. Here's what happens when you call `model.transcribe()`:

```
                         WHISPER INTERNAL PIPELINE
                         
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Audio   │───►│  Mel         │───►│  Encoder     │───►│  Decoder     │
│  (waveform)  │    │  Spectrogram │    │  (Transformer│    │  (Transformer│
│              │    │  (80 bins)   │    │   Blocks)    │    │   Blocks)    │
│  float32[]   │    │  80 × 3000   │    │              │    │              │
│  @ 16kHz     │    │  matrix      │    │  → hidden    │    │  → tokens    │
│              │    │              │    │    states     │    │  → text      │
└─────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
     │                   │                  │                    │
     ▼                   ▼                  ▼                    ▼
  "Hello world"       "Frequency        "Understands         "Outputs text
   as a wave          view of the        WHAT was said        with timing
   of pressure"       sound — like       and WHEN"            information"
                      a piano roll"
```

**Step by step:**

1. **Resampling** — Audio is resampled to 16kHz mono. Why 16kHz? Human speech is bandlimited to ~8kHz (Nyquist says you need 2× → 16kHz). Higher sample rates waste compute on frequencies that don't contain speech.

2. **Mel Spectrogram** — The waveform is converted into a 2D image-like representation using Short-Time Fourier Transform (STFT) + Mel filter bank. The result is an 80×3000 matrix (80 frequency bins × 30 seconds of audio). This is why Whisper processes audio in 30-second chunks.

3. **Encoder** — A stack of Transformer blocks processes the spectrogram and produces hidden states that encode *what was said*. This is analogous to how BERT encodes text — the encoder builds a contextual representation.

4. **Decoder** — An autoregressive Transformer generates text tokens one at a time, conditioned on the encoder output. It also produces timestamp tokens (`<|0.00|>`, `<|0.50|>`, etc.) that tell you *when* each word was spoken.

5. **Beam Search** — The `beam_size=5` parameter in your code means the decoder maintains 5 parallel hypothesis paths and picks the most probable one. Higher beam = more accurate but slower.

### Your Data Flow — Traced Line by Line

Here's the exact data flow through your system when a video is transcribed:

```
main.py::main()
  │
  ├── argparse parses CLI args
  │     url = "https://youtube.com/watch?v=..."
  │     num_shorts = 3
  │
  └── pipeline.py::run(url, num_shorts)
        │
        ├── Stage 1: downloader.py::download_video(url)
        │     └── Returns: DownloadResult(video_path="/downloads/video.mp4",
        │                                 duration=1847.0)
        │
        ├── Stage 2: transcriber_local.py::transcribe(video_path)
        │     │
        │     ├── _extract_audio(video_path)
        │     │     ├── Opens .mp4 with moviepy
        │     │     ├── Extracts audio track
        │     │     ├── Writes to temp .mp3 file
        │     │     └── Returns: "/tmp/tmpXXXXXX.mp3"
        │     │
        │     ├── WhisperModel("base", device="cuda", compute_type="float16")
        │     │     └── Loads ~140MB model weights into VRAM
        │     │
        │     ├── model.transcribe(audio_path, beam_size=5)  ← GENERATOR
        │     │     └── Yields segments lazily as they're decoded
        │     │
        │     ├── Iterates segments → builds TranscriptSegment[]
        │     │
        │     ├── Assembles Transcript(full_text, segments, language, duration)
        │     │
        │     ├── Deletes temp audio file
        │     │
        │     └── Returns: Transcript
        │
        ├── Stage 3: analyzer_local.py::find_best_segments(transcript)
        │     └── ... (covered in a future deep-dive)
        │
        └── Stage 4: editor.py::create_short(...)
              └── ... (covered in a future deep-dive)
```

### The Two Transcription Backends — Cloud vs Local

Your codebase implements a **strategy pattern** — same interface, two implementations:

| Aspect | `transcriber.py` (Cloud) | `transcriber_local.py` (Local) |
|--------|-------------------------|-------------------------------|
| **Engine** | OpenAI Whisper API | faster-whisper (CTranslate2) |
| **Model Loading** | None (server-side) | Loaded into GPU VRAM |
| **Cost** | $0.006/min of audio | $0 (GPU time only) |
| **Latency** | Network round-trip + server queue | ~10-20× faster than real-time on T4 |
| **Accuracy** | whisper-1 (server-optimized) | Depends on model size chosen |
| **Hardware** | No GPU needed locally | Requires CUDA-capable GPU |
| **Privacy** | Audio sent to OpenAI servers | Audio stays on your machine |
| **File Size Limit** | 25MB per request | No limit |
| **Output Format** | `verbose_json` with segments | Generator of segments |
| **Error Handling** | API errors (rate limits, 413) | OOM, model load failures |

The **switching logic** lives in `pipeline.py`:

```python
# pipeline.py lines 73-78 — The strategy switch
if MODE == "local":
    from src.transcriber_local import transcribe          # ← lazy import
    transcript = transcribe(result.video_path, model_size=WHISPER_LOCAL_MODEL)
else:
    from src.transcriber import transcribe                # ← lazy import
    transcript = transcribe(result.video_path)
```

**Why lazy imports?** The `from src.transcriber_local import transcribe` inside the `if` block means `faster_whisper` and its CUDA dependencies are *never imported* if you're in OpenAI mode. This is an intentional optimization — importing `faster_whisper` loads CTranslate2 bindings, which takes >1 second and fails if CUDA isn't installed. Lazy import avoids that failure in cloud mode.

---

## 4. Industry Usage

### How Big Tech Companies Build Transcription Systems

**YouTube (Google)**
- Processes ~500 hours of video uploaded every minute
- Uses a custom ASR (Automatic Speech Recognition) pipeline, not Whisper
- Transcriptions are used for: auto-captions, search indexing, content moderation, ad placement
- They run **streaming ASR** — transcription starts while the video is still uploading
- Results are stored in Bigtable, indexed for full-text search

**Spotify**
- Transcribes all podcasts for search and discovery
- Uses a combination of ASR models specialized for different languages
- Transcription feeds into their recommendation engine

**Zoom / Teams / Google Meet**
- Real-time streaming transcription (completely different architecture from batch)
- Uses WebSocket connections with chunked audio streaming
- Latency requirement: <300ms end-to-end
- Runs on specialized TPUs/GPUs with dedicated inference servers

**Rev.ai / AssemblyAI / Deepgram**
- Transcription-as-a-Service companies
- They run model serving infrastructure (Triton, TorchServe) behind load balancers
- Offer both batch and streaming APIs
- Charge per-minute of audio

### Common Production Patterns

```
                    PRODUCTION TRANSCRIPTION ARCHITECTURE
                    
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│   Object     │    │   Task       │    │   GPU        │    │   Result     │
│   Storage    │───►│   Queue      │───►│   Workers    │───►│   Store      │
│  (S3/GCS)   │    │ (Redis/SQS) │    │ (K8s pods)  │    │ (Postgres/   │
│             │    │             │    │             │    │  Elasticsearch)│
│  audio.mp3  │    │ {job_id,    │    │  Whisper     │    │  {segments,  │
│             │    │  s3_key,    │    │  inference   │    │   full_text, │
│             │    │  priority}  │    │             │    │   language}  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘
       │                                    │
       │                                    │
       ▼                                    ▼
  Webhook/API                         GPU autoscaling:
  callback when                       Scale to 0 when idle,
  complete                            scale up on queue depth
```

**Key patterns:**
1. **Decouple upload from processing** — Upload goes to object storage, processing happens async
2. **Job queue** — Redis/SQS/Celery for job management, retry, and priority
3. **GPU auto-scaling** — GPUs are expensive ($1-3/hr). Scale to zero when idle.
4. **Webhook callbacks** — Don't make the client poll. Push results when done.

### What Is Avoided in Production and Why

| Avoided Pattern | Why |
|-----------------|-----|
| Synchronous transcription in API request | GPU inference takes 30-300s. HTTP requests time out at 30s. Always async. |
| Loading model per-request | Model loading takes 3-10s. Load once, keep in memory, serve many requests. |
| Storing audio on local disk | Single point of failure. Use object storage (S3) for durability. |
| Processing without a queue | No retry on failure, no backpressure, no priority, no monitoring. |
| Using `float32` compute type | Doubles VRAM usage vs `float16`. No accuracy benefit on current GPUs. |

---

## 5. Tradeoffs

### Local Whisper vs Cloud API — The Real Decision Matrix

```
                         COST vs. QUALITY vs. CONTROL
                         
   High Quality ─────────┤
                          │    OpenAI API        Local large-v3
                          │    (whisper-1)       (5× slower, free)
                          │
                          │
                          │    Local small        Local medium
                          │    (fastest, free)    (balanced)
                          │
   Low Quality  ──────────┤
                          │    Local tiny
                          │    (real-time speed)
                          │
                          └─────────────────────────────────────
                        $0/min                             $0.006/min
                        (GPU required)                     (no GPU needed)
```

### Model Size Tradeoffs (faster-whisper)

| Model | VRAM | Speed (vs real-time) | WER (English) | Your Use Case? |
|-------|------|---------------------|---------------|----------------|
| `tiny` | ~1GB | ~32× | ~7.6% | Testing/dev only |
| `base` | ~1GB | ~16× | ~5.0% | ✅ **Your default** — good balance |
| `small` | ~2GB | ~6× | ~3.4% | Better accuracy, still fast on T4 |
| `medium` | ~5GB | ~2× | ~2.9% | Diminishing returns |
| `large-v3` | ~10GB | ~1× | ~2.0% | Overkill for clip selection |

**Why `base` is the right choice for your system:** Your transcription feeds into an LLM that picks *segments*. It doesn't need word-perfect accuracy — it needs to understand the *gist* of what was said in each time window. A 5% WER means ~1 in 20 words is wrong. For segment selection, that's fine. You'd only need `large-v3` if you were doing medical/legal transcription where every word matters.

### Audio Format Tradeoffs

| Format | Why Your Code Uses It | Alternative | Why Not |
|--------|----------------------|-------------|---------|
| `.mp3` (lossy) | Small files, faster I/O | `.wav` (lossless) | 10× larger files, no accuracy benefit for speech |
| `16kHz` sample rate | Whisper's native rate | `44.1kHz` | Whisper downsamples anyway — wasted compute |
| Mono | Speech is mono | Stereo | Doubles data, no benefit (speech is mixed to mono internally) |

### `float16` vs `float32` vs `int8` Compute

```python
# Your code: transcriber_local.py line 54
model = WhisperModel(model_size, device="cuda", compute_type="float16")
```

| Compute Type | VRAM Usage | Speed | Accuracy |
|-------------|-----------|-------|----------|
| `float32` | 2× VRAM | Baseline | Baseline |
| `float16` | 1× VRAM | ~2× faster | ~Identical | ← **Your choice**
| `int8` | 0.5× VRAM | ~3× faster | Slight degradation |

**Why `float16`:** On modern GPUs (T4, A100), the Tensor Cores are optimized for `float16` math. You get 2× throughput for free with no measurable accuracy loss. `int8` gives another 50% speedup but requires quantization-aware tuning and can degrade on edge cases (accented speech, fast talking). `float16` is the industry standard safe default.

---

## 6. Building Perspective

### Where This Fits in Your AI Agent System

```
┌────────────────────────────────────────────────────────────────────┐
│                        YOUR SYSTEM                                 │
│                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │          │    │              │    │              │              │
│  │ Download │    │ TRANSCRIBE   │    │  AI Analyze  │              │
│  │ (yt-dlp) │───►│              │───►│ (LLM picks   │              │
│  │          │    │  The CRITICAL│    │  segments)   │              │
│  └──────────┘    │  DATA BRIDGE │    └──────┬───────┘              │
│                  │              │           │                      │
│                  │  Video → Text│           │                      │
│                  │  + Timestamps│           ▼                      │
│                  │              │    ┌──────────────┐              │
│                  │              │───►│              │              │
│                  └──────────────┘    │  Video Edit  │              │
│                                     │ (uses timestamps            │
│                                     │  for captions +             │
│                                     │  clip cutting)              │
│                                     │              │              │
│                                     └──────────────┘              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The transcription output is consumed by **two downstream systems**:

1. **The Analyzer** — reads `Transcript.full_text` and `segments` to decide WHICH clips to cut
2. **The Editor** — reads `Transcript.segments` to generate word-by-word animated captions with precise timing

If transcription is wrong → analyzer picks bad segments → editor renders wrong captions → output is garbage. **Transcription is the foundation of everything.**

### How It Would Interact With Other Systems (If You Scale)

If you were building this at company scale, the transcription module would connect to:

```
Client App  ──HTTP POST──►  API Gateway  ──Publish──►  Message Queue
                                                          │
                                                          ▼
Webhook ◄── Result Store ◄── GPU Workers (transcription)
                     │
                     ▼
               Search Index (Elasticsearch)
               AI Analysis Pipeline (downstream)
               Caption Generator (downstream)
               Content Moderation (downstream)
```

---

## 7. Common Mistakes

### Mistake 1: Not Handling the Generator Correctly

```python
# YOUR CODE (transcriber_local.py line 58):
segments_gen, info = model.transcribe(audio_path, beam_size=5)
```

**What most people miss:** `model.transcribe()` returns a **generator**, not a list. The transcription happens lazily as you iterate. If you call `len(segments_gen)` or try to iterate it twice, it fails silently or gives empty results.

Your code handles this correctly by iterating once and building a list:

```python
# transcriber_local.py lines 61-69 — CORRECT: Consume generator once
segments = []
full_text_parts = []
for seg in segments_gen:                     # ← Iterates the generator exactly once
    segments.append(TranscriptSegment(
        start=seg.start,
        end=seg.end,
        text=seg.text.strip(),
    ))
    full_text_parts.append(seg.text.strip())
```

**The bug juniors write:**
```python
# ❌ WRONG: This silently returns 0 because generators have no len()
count = len(segments_gen)

# ❌ WRONG: Second iteration is empty — generator already consumed
for seg in segments_gen: process(seg)
for seg in segments_gen: display(seg)  # ← This loop does nothing
```

### Mistake 2: Not Cleaning Up Temp Files on Failure

```python
# YOUR CODE (transcriber_local.py lines 78-79):
# Cleanup
Path(audio_path).unlink(missing_ok=True)
```

**The problem:** If `model.transcribe()` throws an exception (OOM, corrupted audio), this cleanup line never executes. Temp files accumulate on disk.

**Production fix:**
```python
# Better: Use try/finally or context manager
try:
    audio_path = _extract_audio(video_path)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments_gen, info = model.transcribe(audio_path, beam_size=5)
    # ... process segments ...
finally:
    Path(audio_path).unlink(missing_ok=True)   # Always runs, even on exception
```

### Mistake 3: Hardcoding `device="cuda"`

```python
# YOUR CODE (transcriber_local.py line 54):
model = WhisperModel(model_size, device="cuda", compute_type="float16")
```

This crashes instantly on any machine without an NVIDIA GPU. In production:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"  # CPU doesn't support float16

model = WhisperModel(model_size, device=device, compute_type=compute_type)
```

### Mistake 4: Ignoring the 25MB API File Size Limit

```python
# YOUR CODE (transcriber.py lines 58-64):
with open(audio_path, "rb") as audio_file:
    response = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
```

If the video is long (>1 hour), the extracted `.mp3` can exceed 25MB — the OpenAI API hard limit. It'll return a `413 Request Entity Too Large` error. **No chunking logic exists in your code.** This is a latent bug.

**Production fix:**
```python
def _chunk_audio(audio_path: str, max_size_mb: int = 24) -> list[str]:
    """Split audio into chunks under the API size limit."""
    file_size = Path(audio_path).stat().st_size
    if file_size <= max_size_mb * 1024 * 1024:
        return [audio_path]
    
    # Use pydub or ffmpeg to split by duration
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(audio_path)
    chunk_duration_ms = int(len(audio) * (max_size_mb * 1024 * 1024 / file_size))
    
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_path = f"/tmp/chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    
    return chunks
```

### Mistake 5: Duplicated Data Classes

Both `transcriber.py` and `transcriber_local.py` define **identical** `TranscriptSegment` and `Transcript` classes. This violates DRY and creates a maintenance hazard — if you add a field to one, you must remember to update the other.

**Production fix:**
```python
# models.py — Single source of truth
@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

@dataclass
class Transcript:
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
```

Then both transcribers import from `models.py`.

---

## 8. Engineering Best Practices

### Clean Architecture Approach

Your current code is a **well-structured modular pipeline**. Here's how to make it production-grade:

```
src/
├── models.py                    # ← NEW: Shared data models (Transcript, Segment)
├── config.py                    # ✅ Already centralized
├── transcriber/                 # ← Refactor into package
│   ├── __init__.py              # Exports: transcribe()
│   ├── base.py                  # Abstract TranscriberBase
│   ├── cloud.py                 # OpenAI implementation
│   ├── local.py                 # faster-whisper implementation
│   └── factory.py               # get_transcriber(mode) → TranscriberBase
├── analyzer/
│   ├── __init__.py
│   ├── base.py
│   ├── openai_analyzer.py
│   └── groq_analyzer.py
├── downloader.py                # ✅ Already clean
├── editor.py                    # ✅ Already clean
└── pipeline.py                  # Orchestrator
```

**The abstract base (Strategy Pattern):**

```python
# src/transcriber/base.py
from abc import ABC, abstractmethod
from src.models import Transcript

class TranscriberBase(ABC):
    """Abstract base for all transcriber implementations."""
    
    @abstractmethod
    def transcribe(self, video_path: str) -> Transcript:
        """Transcribe a video file and return structured transcript."""
        pass
    
    def _extract_audio(self, video_path: str) -> str:
        """Shared audio extraction logic."""
        # ... same code, defined ONCE
```

**The factory:**

```python
# src/transcriber/factory.py
def get_transcriber(mode: str) -> TranscriberBase:
    if mode == "local":
        from .local import LocalTranscriber
        return LocalTranscriber()
    else:
        from .cloud import CloudTranscriber
        return CloudTranscriber()
```

### Testing Strategy

```python
# tests/test_transcriber.py
import pytest
from unittest.mock import patch, MagicMock
from src.transcriber_local import transcribe, _extract_audio, TranscriptSegment

class TestExtractAudio:
    """Test audio extraction independently from transcription."""
    
    def test_produces_mp3_file(self, tmp_path):
        """Audio extraction should produce a valid .mp3 file."""
        # Arrange: create a test video with moviepy
        # Act: extract audio
        # Assert: file exists, has .mp3 extension, size > 0
    
    def test_cleanup_on_error(self):
        """Temp file should be cleaned up even if extraction fails."""

class TestTranscribe:
    """Test transcription with mocked Whisper model."""
    
    @patch('src.transcriber_local.WhisperModel')
    def test_returns_transcript_with_segments(self, mock_model):
        """Transcription should return Transcript with proper segments."""
        # Arrange: mock model.transcribe() to return fake segments
        mock_seg = MagicMock(start=0.0, end=5.0, text=" Hello world ")
        mock_model.return_value.transcribe.return_value = (
            iter([mock_seg]),      # generator
            MagicMock(language="en")  # info
        )
        
        # Act
        result = transcribe("fake_video.mp4", model_size="tiny")
        
        # Assert
        assert result.full_text == "Hello world"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"  # stripped
        assert result.language == "en"
    
    def test_empty_video_returns_empty_transcript(self):
        """Video with no speech should return empty Transcript."""

    def test_long_video_handles_many_segments(self):
        """Ensure memory doesn't explode on 2-hour videos."""
```

### Monitoring & Observability

In production, you need to know:

```python
import time
import logging

logger = logging.getLogger(__name__)

def transcribe(video_path: str, model_size: str = "base") -> Transcript:
    start_time = time.monotonic()
    
    # ... transcription logic ...
    
    elapsed = time.monotonic() - start_time
    audio_duration = transcript.duration
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    
    logger.info(
        "transcription_complete",
        extra={
            "model_size": model_size,
            "audio_duration_s": audio_duration,
            "wall_time_s": elapsed,
            "real_time_factor": rtf,        # <1 means faster than real-time
            "segment_count": len(transcript.segments),
            "language": transcript.language,
            "words_per_minute": len(transcript.full_text.split()) / (audio_duration / 60),
        }
    )
```

**Key metric: Real-Time Factor (RTF)** — If RTF = 0.1, it takes 6 seconds to transcribe 60 seconds of audio. If RTF > 1.0, your system is slower than real-time and needs a bigger GPU or smaller model.

### Scaling Considerations

| Scale | Architecture |
|-------|-------------|
| **1 video/day (you)** | Single process, local GPU. Your current setup. |
| **100 videos/day** | Celery + Redis queue, 1-2 GPU workers. |
| **10K videos/day** | Kubernetes GPU pods, auto-scaling based on queue depth. |
| **1M+ videos/day (YouTube)** | Custom ASR, TPU clusters, streaming pipeline, sharded processing. |

Your architecture is correct for its scale. Don't over-engineer.

---

## 9. Learning Path

### What to Learn Next (In Order)

```
                    YOUR LEARNING DEPENDENCY GRAPH
                    
You Are Here ───► Transcription Pipeline Architecture
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Audio Signal    Model Serving    Queue-Based
    Processing      (Triton/        Architecture
    (DSP basics,    TorchServe)     (Celery/Redis)
     FFmpeg)              │              │
          │              ▼              ▼
          ▼         GPU Compute     Distributed
    Streaming       (CUDA basics,   Systems
    ASR             memory mgmt)    (task routing,
    (WebSockets,         │          retry, DLQ)
     chunked             ▼
     audio)         ML Optimization
                    (quantization,
                     distillation,
                     ONNX export)
```

### Recommended Next Topics for Your System

1. **Audio Signal Processing with FFmpeg** — Your `_extract_audio` uses moviepy (which wraps FFmpeg). Learning FFmpeg directly gives you: audio normalization, silence detection, loudness metering, format conversion. All critical for production audio pipelines.

2. **LLM Prompt Engineering for Structured Output** — Your analyzer sends transcripts to GPT/Llama3 and parses JSON. Understanding tokenization, context window management, and structured output (JSON mode, function calling) directly improves your segment selection quality.

3. **Task Queue Architecture** — You already know Celery+Redis from your EDR project. Applying it here means: upload video → return job_id → process async → webhook on completion. This is the #1 change to make your system production-ready.

4. **Video Processing Pipeline (FFmpeg + GPU encoding)** — Your editor uses moviepy, which is slow for encoding. Learning GPU-accelerated encoding (NVENC) and FFmpeg's filter graph gives you 5-10× faster video export.

---

## 10. Real Example — Your Codebase Dissected

### Line-by-Line Walkthrough: `transcriber_local.py`

```python
"""                                          # lines 1-4: MODULE DOCSTRING
Local Transcriber — uses faster-whisper      # ──────────────────────────────────
(CTranslate2) for GPU-accelerated            # faster-whisper is NOT the original
transcription. Free, no API key needed.      # OpenAI whisper. It's a C++ re-
"""                                          # implementation using CTranslate2
                                             # that is 4× faster with identical
                                             # accuracy. This matters.
```

```python
from dataclasses import dataclass, field     # line 6: Python's built-in struct
import tempfile                              # line 7: OS-managed temp files 
from pathlib import Path                     # line 8: Cross-platform path handling

from moviepy import VideoFileClip            # line 10: Video file I/O
from rich.console import Console             # line 11: Terminal pretty-printing

console = Console()                          # line 13: Global console instance
```

**Design decision — why `dataclass` and not Pydantic?**

`dataclass` is stdlib, zero dependencies. For a data transfer object (DTO) that's only used internally, it's the right choice. Pydantic adds validation, serialization, and schema generation — useful for API boundaries, overkill for internal pipeline structs.

```python
@dataclass
class TranscriptSegment:                     # lines 16-21
    """A single segment of the transcript."""
    start: float    # Whisper's segment start time in seconds
    end: float      # Whisper's segment end time in seconds  
    text: str       # The actual words spoken in this time window
```

**Why this struct exists:** Whisper outputs segments as its own internal objects. By mapping them to your own `TranscriptSegment`, you **decouple** your pipeline from Whisper's API. If you switch to a different ASR engine tomorrow, only this mapping code changes — not the analyzer or editor.

```python
@dataclass
class Transcript:                            # lines 24-30
    full_text: str                           # Joined from all segments
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""                       # Detected language code ("en", "es")
    duration: float = 0.0                    # Total audio duration
```

**Why `field(default_factory=list)` instead of `= []`?**

Classic Python gotcha. If you write `segments: list = []`, ALL instances of `Transcript` share the SAME list object (mutable default argument bug). `field(default_factory=list)` creates a NEW list for each instance. This is a correctness issue, not style.

```python
def transcribe(video_path: str, model_size: str = "base") -> Transcript:
    # ...
    from faster_whisper import WhisperModel  # line 45: LAZY IMPORT
```

**Why lazy import inside the function?**

Three reasons:
1. `faster_whisper` imports CTranslate2 C++ bindings → takes ~1-2 seconds to import
2. If CUDA isn't installed, the import crashes → this crash only happens when you actually try to transcribe locally, not when importing the module
3. Cloud mode never needs this import → zero waste

This is a **conditional dependency** pattern. Common in Python projects that support multiple backends.

```python
    # Step 1: Extract audio
    audio_path = _extract_audio(video_path)  # line 50
```

**Why extract audio first instead of giving the video directly to Whisper?**

Whisper models accept audio files, not video files. A `.mp4` file contains both video streams and audio streams in a container format. MoviePy's `VideoFileClip.audio.write_audiofile()` uses FFmpeg under the hood to demux the audio stream and encode it as MP3. This is the same as running `ffmpeg -i video.mp4 -vn -acodec libmp3lame audio.mp3`.

```python
    # Step 2: Load model on GPU
    model = WhisperModel(model_size, device="cuda", compute_type="float16")  # line 54
```

**What happens on this line internally:**

1. Downloads model weights from HuggingFace (first time only) → cached in `~/.cache/huggingface/`
2. Loads weights into CPU RAM (~140MB for `base`)
3. Transfers weights to GPU VRAM
4. Compiles the CTranslate2 computation graph
5. Pre-allocates working buffers for inference

**This takes 2-5 seconds.** In a production system, you'd load once and reuse:

```python
# Production pattern: singleton model
_model_cache = {}

def get_model(model_size: str) -> WhisperModel:
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(model_size, device="cuda", compute_type="float16")
    return _model_cache[model_size]
```

```python
    # Step 3: Transcribe
    segments_gen, info = model.transcribe(audio_path, beam_size=5)  # line 58
```

**What `beam_size=5` means:**

During decoding, the model generates text token by token. At each step, instead of greedily picking the most probable next token, it maintains 5 parallel hypotheses (beams) and expands each one. At the end, it selects the hypothesis with the highest total probability.

```
Step 1:  "The"(0.9)  "A"(0.05)  "This"(0.03)  ...  [keep top 5]
Step 2:  "The cat"(0.7)  "The dog"(0.15)  "A cat"(0.03)  ...
Step 3:  "The cat sat"(0.6)  "The cat is"(0.08)  ...
...
Final:   "The cat sat on the mat" ← highest total probability
```

`beam_size=5` is the sweet spot. `beam_size=1` is greedy (fast, less accurate). `beam_size=10`+ is diminishing returns (slower, barely more accurate).

```python
    # Step 4: Collect segments
    segments = []
    full_text_parts = []
    for seg in segments_gen:                      # line 63: consuming the generator
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),                # line 67: strip whitespace
        ))
        full_text_parts.append(seg.text.strip())
```

**Why `.strip()`?** Whisper often prepends a space to segment text (e.g., `" Hello world"` instead of `"Hello world"`). This is because the Whisper tokenizer includes a leading space as part of word tokens (similar to GPT's tokenizer). Stripping ensures clean text for downstream processing.

```python
    transcript = Transcript(
        full_text=" ".join(full_text_parts),      # line 72: join into one string
        segments=segments,
        language=info.language,                   # line 74: auto-detected language
        duration=segments[-1].end if segments else 0.0,  # line 75
    )
```

**Line 75 — `segments[-1].end if segments else 0.0`:**

The total audio duration isn't directly returned by faster-whisper's `info` object (unlike the OpenAI API). So the code uses the end time of the last segment as an approximation. This is slightly inaccurate — there might be 2-3 seconds of silence after the last spoken word — but it's good enough for segment selection.

```python
    # Cleanup
    Path(audio_path).unlink(missing_ok=True)      # line 79: delete temp audio
```

**`missing_ok=True`** — If the file was already deleted (e.g., by the OS or a concurrent cleanup), don't throw an error. Defensive programming.

### Line-by-Line Walkthrough: `transcriber.py` (Cloud Version)

The cloud version follows the same pattern but with key differences:

```python
from openai import OpenAI                        # line 10
from src.config import OPENAI_API_KEY, WHISPER_MODEL  # line 13

# ...

client = OpenAI(api_key=OPENAI_API_KEY)          # line 56

with open(audio_path, "rb") as audio_file:       # line 58
    response = client.audio.transcriptions.create(
        model=WHISPER_MODEL,                     # "whisper-1"
        file=audio_file,                         # raw bytes, uploaded
        response_format="verbose_json",          # line 62
        timestamp_granularities=["segment"],     # line 63
    )
```

**Why `response_format="verbose_json"`?**

OpenAI's Whisper API supports multiple output formats:

| Format | What You Get | When to Use |
|--------|-------------|-------------|
| `text` | Plain text only | Just need the words, no timestamps |
| `json` | `{"text": "..."}` | Basic text with metadata |
| `verbose_json` | Text + segments + word timestamps | ✅ **Your choice** — need timestamps for clip cutting |
| `srt` | Subtitle format | Directly generating subtitle files |
| `vtt` | WebVTT subtitle format | Web player captions |

You need `verbose_json` because your analyzer and editor both need timestamp information.

**Why `timestamp_granularities=["segment"]`?**

This tells the API to return segment-level timestamps (5-30 second chunks). The alternative is `["word"]` which gives word-level timestamps — useful for karaoke-style captions but slower and more expensive. Your current caption system uses segment-level timing, so `"segment"` is correct.

```python
# lines 68-73: Parse response — API returns dict-like objects
for seg in response.segments:
    segments.append(TranscriptSegment(
        start=seg["start"],           # ← Dict access (API returns dicts)
        end=seg["end"],
        text=seg["text"].strip(),
    ))
```

**Notice the difference from local:** In `transcriber_local.py`, segments have attributes (`seg.start`). In `transcriber.py`, segments are dicts (`seg["start"]`). This is because faster-whisper returns named objects while the OpenAI API returns JSON parsed into dicts. This asymmetry is another reason to have a shared data model — the mapping logic handles the difference.

---

### Complete Data Flow Diagram

```
USER RUNS:
  python main.py "https://youtube.com/watch?v=abc123" --shorts 3

     │
     ▼
main.py: argparse → run(url="...", num_shorts=3)
     │
     ▼
pipeline.py: run()
     │
     ├──► download_video(url)
     │       │
     │       ├── yt-dlp extracts video metadata (title, duration, thumbnail)
     │       ├── Downloads best quality mp4 (video + audio streams)
     │       ├── Merges streams with FFmpeg (if separate)
     │       └── Returns: DownloadResult(video_path="downloads/My Video.mp4",
     │                                   duration=1847.0)
     │
     ├──► transcribe(video_path="downloads/My Video.mp4", model_size="base")
     │       │
     │       ├── _extract_audio("downloads/My Video.mp4")
     │       │       │
     │       │       ├── moviepy opens MP4 container
     │       │       ├── FFmpeg extracts audio stream
     │       │       ├── Encodes to MP3 (lossy, small file)
     │       │       └── Writes to: /tmp/tmpABC123.mp3
     │       │
     │       ├── WhisperModel("base", device="cuda", compute_type="float16")
     │       │       │
     │       │       ├── Loads 140MB model into GPU VRAM
     │       │       └── Pre-allocates inference buffers
     │       │
     │       ├── model.transcribe("/tmp/tmpABC123.mp3", beam_size=5)
     │       │       │
     │       │       ├── Resamples audio to 16kHz mono
     │       │       ├── Computes 80-bin Mel spectrogram
     │       │       ├── Processes in 30s chunks through encoder
     │       │       ├── Beam search decoder generates text + timestamps
     │       │       └── Yields segments as generator
     │       │
     │       ├── Consumes generator → builds TranscriptSegment[]
     │       │       segment[0]: {start: 0.0,  end: 4.2, text: "Today we're going to..."}
     │       │       segment[1]: {start: 4.2,  end: 8.1, text: "talk about something..."}
     │       │       segment[2]: {start: 8.1,  end: 12.5, text: "that will change..."}
     │       │       ...
     │       │       segment[N]: {start: 1840.0, end: 1847.0, text: "thanks for watching"}
     │       │
     │       ├── Assembles Transcript:
     │       │       full_text: "Today we're going to talk about something..."
     │       │       segments:  [TranscriptSegment × 423]
     │       │       language:  "en"
     │       │       duration:  1847.0
     │       │
     │       ├── Deletes /tmp/tmpABC123.mp3
     │       └── Returns: Transcript
     │
     ├──► find_best_segments(transcript, num_shorts=3, video_duration=1847.0)
     │       │
     │       ├── Formats transcript as timestamped text:
     │       │       "[0:00 → 0:04] Today we're going to..."
     │       │       "[0:04 → 0:08] talk about something..."
     │       │       ...
     │       │
     │       ├── Smart-truncates if > 7000 chars (Groq token limit)
     │       ├── Sends to LLM (Groq/GPT) for segment selection
     │       ├── Parses JSON response → validates timestamps
     │       ├── Removes overlapping segments  
     │       └── Returns: [Segment × 3]
     │
     └──► create_short(video_path, segment, transcript, index)  × 3
             │
             ├── Crops video to 9:16 (portrait)
             ├── Applies zoom effect
             ├── Renders word-by-word captions using transcript timestamps
             ├── Adds hook card overlay
             ├── Mixes background music
             ├── Encodes with libx264 @ 5000k bitrate
             └── Exports: output/short_1.mp4, short_2.mp4, short_3.mp4
```

---

## Summary

| Concept | Your Implementation | Production Upgrade |
|---------|--------------------|--------------------|
| **Transcription Engine** | faster-whisper (local) / OpenAI API (cloud) | Add model caching, GPU auto-detection |
| **Audio Extraction** | moviepy → temp .mp3 | Direct FFmpeg for speed, temp file cleanup in `finally` |
| **Data Models** | Duplicated in both files | Single `models.py` shared module |
| **Error Handling** | Minimal | Retry with backoff, structured logging, cleanup guarantees |
| **File Handling** | Temp files, manual cleanup | Context managers, `/finally` blocks |
| **Architecture** | If/else mode switch | Strategy pattern with abstract base |
| **Scaling** | Single process | Celery + Redis queue → GPU workers |
| **Testing** | None | Mock-based unit tests, integration tests with tiny model |
| **Monitoring** | Console prints | Structured logging, RTF metrics, Prometheus/Grafana |

Your codebase is **solidly structured for a prototype**. The modular design is correct. The main improvements are: defensive error handling, deduplication of data models, and preparation for async processing.

---

> **Next**: Ready for Deep Dive #2? Options:
> - **AI Agent Orchestration** — how the analyzer decides which segments to pick
> - **Media Processing Pipeline** — how the editor renders the final shorts
> - **Queue-Based Architecture** — how to make this system handle 100 videos/day
