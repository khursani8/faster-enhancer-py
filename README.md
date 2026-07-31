# faster-enhancer-py

A Python binding to **[faster-enhancer.c](https://github.com/kdrkdrkdr/faster-enhancer.c)** —
a dependency-free, streaming 48 kHz int8 (W8A8) speech-denoiser engine that
runs in a fraction of one CPU core.

This package is a thin **ctypes frontend** over the compiled C engine: the C
library does all the inference work, so you get **native speed** and
**bit-identical output** to the C runtime. No inference framework, no heap
allocation after startup, no retraining. The engine runs the published
FastEnhancer-Medium weights unchanged.

The design and measurements behind the engine are in
[arXiv:2607.25350](https://arxiv.org/abs/2607.25350).

```
320 samples in → 320 samples out   (6.67 ms frame, 48 kHz, no look-ahead)
~0.5 ms / frame on a modern x86/ARM core     (real-time factor < 0.1)
565 KB W8A8 weight blob
```

---

## Install

Prebuilt wheels are published for Linux (x86_64, aarch64) and macOS (x86_64,
arm64):

```sh
pip install faster-enhancer-py
```

Users on those platforms never need a C compiler — the wheel carries the
compiled `libfe` shared library and the `fe.q8` weights.

### Building from source

If no wheel matches your platform (e.g. Windows), build from an sdist — you
need a C compiler (GCC or Clang; **MSVC is unsupported** by the engine) and
CMake ≥ 3.20:

```sh
pip install faster-enhancer-py --no-binary faster-enhancer-py
# or, from a checkout:
pip install .
```

The build vendors the C sources under `csrc/` and compiles `libfe` as a shared
library that the Python layer loads via ctypes.

---

## Quick start

```python
from faster_enhancer import Enhancer, FRAME_SIZE, SAMPLE_RATE

enh = Enhancer()                       # loads the vendored fe.q8 weights
for chunk in your_48kHz_stream(FRAME_SIZE):
    enhanced = enh.run(chunk)          # float32[320] in, float32[320] out
enh.close()
```

`Enhancer` is also a context manager (`with Enhancer() as enh: ...`).

### What `run()` expects

- **One call = `FRAME_SIZE` (320) samples** at 48 kHz. The call emits 320
  samples; there is a fixed 704-sample (14.67 ms) STFT alignment delay before
  the enhanced signal aligns with the input.
- Streaming state (GRU hidden state + STFT/iSTFT overlap-add caches) is kept
  across calls. Call `enh.reset()` between independent audio streams.
- The C engine owns a **single global instance**; only one `Enhancer` may be
  live at a time. Constructing a second one raises `RuntimeError` until the
  first is `close()`-d. Single-threaded: do not call `run()` concurrently.

### Loading your own weights

```python
Enhancer(open("/path/to/fe.q8", "rb").read())
```

---

## API

| Member | Description |
|---|---|
| `Enhancer(weights_blob=None, *, lib_path=None)` | Create the engine. `weights_blob` defaults to the vendored `fe.q8`. |
| `enh.run(audio_in) -> np.ndarray` | Enhance one 320-sample frame → `float32[320]`. |
| `enh.reset()` | Clear streaming state. |
| `enh.close()` | Release the C engine (also via context manager / `__del__`). |
| `FRAME_SIZE` | `320` samples per `run()`. |
| `SAMPLE_RATE` | `48000` Hz. |
| `load_default_weights()` | Return the vendored `fe.q8` blob bytes. |

---

## Running on a file

An end-to-end CLI example is in `examples/enhance_file.py`. Feed raw 48 kHz
mono float32, get raw 48 kHz mono float32 out:

```sh
python examples/enhance_file.py noisy.f32 --out enhanced.f32
```

(`ffmpeg -ar 48000 -ac 1 -f f32le` produces/consumes that format.)

---

## How it relates to the C engine

This is a binding, not a reimplementation:

- The compiled `libfe` **is** the C engine. `Enhancer.run()` is a direct call
  into `fe_run()` (see `include/fe.h`). Output is therefore bit-identical to
  the C library, including its runtime CPU-feature dispatch (the engine picks
  the best SIMD tier — AVX2 / AVX-VNNI / AVX-512-VNNI on x86, NEON / DOTPROD /
  I8MM on ARM — at initialization and has no scalar fallback).
- There is **no scalar fallback**: a host below the baseline ISA (x86: AVX2 +
  FMA3 + F16C; ARM: NEON) makes `Enhancer(...)` raise `RuntimeError`.

## Acknowledgements & license

The model, architecture, and weights are **FastEnhancer** by Sunghwan Ahn
et al. — see [aask1357/fastenhancer](https://github.com/aask1357/fastenhancer).
The C runtime is **faster-enhancer.c** by Gyeongmin Kim. This package is an
independent Python binding and is not affiliated with either author.

MIT licensed. See [csrc/LICENSE](csrc/LICENSE) and [csrc/NOTICE](csrc/NOTICE).
