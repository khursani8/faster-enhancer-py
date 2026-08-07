# FastEnhancer Telephony Noise Suppression — Deployment Guide

## What This Does

Removes background noise from contact center phone calls to improve downstream ASR accuracy.
- Model: 206K params, 827KB ONNX
- Speed: 8x faster than real-time on CPU
- Output: clean speech, no buzz, VAD-gated silence

## Quick Start

```bash
# 1. Install dependencies
pip install onnxruntime librosa soundfile scipy torch torchaudio numpy

# 2. Enhance a single file
python examples/streaming_enhance.py --input call.wav --output clean.wav --vad

# 3. Batch process a directory
python examples/streaming_enhance.py --input-dir calls/ --output-dir enhanced/ --vad
```

## Python API

```python
import sys
sys.path.insert(0, "examples")
from streaming_enhance import StreamingEnhancer

# Initialize (loads model once)
enhancer = StreamingEnhancer(
    onnx_path="faster_enhancer/weights/fastenhancer.onnx",
    vad=True,           # Silero VAD: suppresses noise in silence
    spectral_gate=False,
    normalize_db=-14.0, # Target loudness
    lowpass_freq=4500,  # Removes BWE artifacts (buzz above 4.5kHz)
)

# Process a file
rtf = enhancer.enhance_file("noisy_call.wav", "clean_call.wav")
print(f"RTF: {rtf:.3f}")  # Should be < 1.0 for real-time

# Or process audio array directly
import librosa
audio, sr = librosa.load("noisy_call.wav", sr=16000)
enhanced = enhancer.enhance_audio(audio)
```

## ASR Integration

```python
import requests, json, tempfile, soundfile as sf

def transcribe(audio_path):
    """Send audio to ASR endpoint."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            "http://YOUR_ASR_ENDPOINT/recognize",
            headers={"Content-Type": "audio/wav"},
            data=f.read()
        )
    return resp.json().get("text", "").strip()

# Pipeline: noisy call → enhance → ASR
enhancer = StreamingEnhancer(vad=True, lowpass_freq=4500)
enhancer.enhance_file("call.wav", "enhanced.wav")

# Send enhanced audio to ASR
transcript = transcribe("enhanced.wav")
print(transcript)
```

## Streaming (Frame-by-Frame)

For real-time call processing:

```python
enhancer = StreamingEnhancer(vad=True, lowpass_freq=4500)
cache = enhancer._init_cache()

# Process 256 samples at a time (16ms at 16kHz)
for chunk in audio_chunks:
    enhanced_frame, cache = enhancer.enhance_frame(chunk, cache)
    # Send enhanced_frame to ASR or playback
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vad` | True | Silero VAD gate (kills noise in silence) |
| `lowpass_freq` | 4500 | Cutoff for BWE buzz removal (Hz) |
| `normalize_db` | -14.0 | Target output loudness (dB) |
| `spectral_gate` | False | Wiener-like spectral noise gate |

## Requirements

- Python 3.10+
- onnxruntime (CPU or GPU)
- librosa, soundfile (audio I/O)
- scipy (lowpass filter)
- torch (for Silero VAD only)
- numpy

## File Locations

- ONNX model: `faster_enhancer/weights/fastenhancer.onnx` (827KB)
- Streaming code: `examples/streaming_enhance.py`
- This guide: `DEPLOYMENT.md`

## Performance

```
Metric              Value
RTF (CPU):          0.03-0.13 (real-time ✅)
Model size:         827KB
Memory:             <500MB
Sample rate:        16kHz mono
Frame size:         256 samples (16ms)
```
