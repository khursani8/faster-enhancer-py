"""faster-enhancer-py: a Python binding to faster-enhancer.c.

A streaming 48 kHz int8 (W8A8) speech-denoiser engine. This package is a thin
ctypes frontend over the C runtime (``faster-enhancer.c``); the compiled C
engine does all the inference work, so output is **bit-identical** to the C
library and runs at native speed (a fraction of one core per stream).

Quick start:
    from faster_enhancer import Enhancer, FRAME_SIZE, SAMPLE_RATE

    enh = Enhancer()                       # loads the vendored fe.q8 weights
    for each 320-sample frame at 48 kHz:
        out = enh.run(noisy_frame)         # float32[320]
    enh.close()                            # (or use: with Enhancer() as enh: ...)

Constants:
    FRAME_SIZE  = 320   samples per run() call (6.67 ms at 48 kHz)
    SAMPLE_RATE = 48000
"""

from __future__ import annotations

from ._binding import (
    DEFAULT_WEIGHTS_PATH,
    Enhancer,
    FRAME_SIZE,
    SAMPLE_RATE,
    load_default_weights,
)

__version__ = "0.1.0"

__all__ = [
    "Enhancer",
    "FRAME_SIZE",
    "SAMPLE_RATE",
    "load_default_weights",
    "DEFAULT_WEIGHTS_PATH",
    "__version__",
]
