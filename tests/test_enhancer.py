"""Unit tests for faster_enhancer.

Self-contained (no external audio files). Skips automatically if the package
or its compiled ``libfe`` cannot be loaded (e.g. a fresh checkout before
``pip install .`` / wheel install).
"""
from __future__ import annotations

import numpy as np
import pytest

# Try to import; skip the whole module if unavailable.
try:
    import faster_enhancer as fe
    from faster_enhancer import Enhancer, FRAME_SIZE, SAMPLE_RATE
    _READY = True
    _REASON = ""
except Exception as e:  # ImportError or OSError from libfe not found
    fe = None
    Enhancer = FRAME_SIZE = SAMPLE_RATE = None
    _READY = False
    _REASON = f"faster_enhancer/libfe not available ({e!r})"

pytestmark = pytest.mark.skipif(not _READY, reason=_REASON)

ALIGN_DELAY = 704  # samples (48 kHz)


# --------------------------------------------------------------------------- #
# API / contract tests.
# --------------------------------------------------------------------------- #
def test_public_constants():
    assert FRAME_SIZE == 320
    assert SAMPLE_RATE == 48000


def test_run_shape_and_dtype():
    with Enhancer() as enh:
        y = enh.run(np.zeros(FRAME_SIZE, dtype=np.float32))
        assert y.shape == (FRAME_SIZE,)
        assert y.dtype == np.float32


def test_silence_in_near_silence_out():
    """After the STFT fill transient, a silent frame produces a near-silent
    output (the engine flushes to zero under FTZ/DAZ)."""
    with Enhancer() as enh:
        enh.run(np.zeros(FRAME_SIZE, dtype=np.float32))  # prime the stream
        y = enh.run(np.zeros(FRAME_SIZE, dtype=np.float32))
        assert np.abs(y).max() < 1e-3


def test_accepts_array_like_and_coerces_float32():
    with Enhancer() as enh:
        y = enh.run([0.0] * FRAME_SIZE)  # list -> float32
        assert y.dtype == np.float32
        assert y.shape == (FRAME_SIZE,)


def test_run_rejects_wrong_length():
    with Enhancer() as enh:
        with pytest.raises(ValueError):
            enh.run(np.zeros(FRAME_SIZE - 1, dtype=np.float32))


def test_reset_is_safe():
    with Enhancer() as enh:
        enh.reset()
        y = enh.run(np.zeros(FRAME_SIZE, dtype=np.float32))
        assert y.shape == (FRAME_SIZE,)


def test_singleton_enforced():
    """The C engine owns one global instance; a second live Enhancer raises."""
    with Enhancer() as first:
        with pytest.raises(RuntimeError):
            Enhancer()
    # After closing, a new one can be constructed.
    with Enhancer():
        pass


# --------------------------------------------------------------------------- #
# Behavioural test: the enhancer must reduce added noise.
# --------------------------------------------------------------------------- #
def _snr(ref: np.ndarray, test: np.ndarray) -> float:
    p_ref = float(np.sum(ref ** 2))
    p_err = float(np.sum((ref - test) ** 2))
    return 10.0 * np.log10(p_ref / (p_err + 1e-20))


def _enhance_signal(enh, noisy: np.ndarray) -> np.ndarray:
    n = (noisy.shape[0] // FRAME_SIZE) * FRAME_SIZE
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, FRAME_SIZE):
        out[i:i + FRAME_SIZE] = enh.run(noisy[i:i + FRAME_SIZE])
    return out


def test_enhancer_improves_snr_on_speech_like_signal():
    """A voiced-harmonic + white-noise signal should come out with higher SNR
    than the noisy input (robust across CPU tiers — only the gain magnitude
    is tier-dependent)."""
    rng = np.random.default_rng(123)
    n_frames = 80
    t = np.arange(n_frames * FRAME_SIZE) / SAMPLE_RATE
    # Voiced-speech-like: fundamental ~180 Hz with three harmonics.
    clean = sum(0.1 * np.sin(2 * np.pi * f * t) for f in (180.0, 360.0, 540.0))
    clean = clean.astype(np.float32)
    noise = (rng.standard_normal(clean.shape) * 0.4).astype(np.float32)
    noisy = clean + noise

    with Enhancer() as enh:
        enhanced = _enhance_signal(enh, noisy)

    d = ALIGN_DELAY
    L = min(enhanced.shape[0] - d, clean.shape[0])
    in_snr = _snr(clean[:L], noisy[:L])
    out_snr = _snr(clean[:L], enhanced[d:d + L])
    assert out_snr > in_snr + 2.0, f"SNR did not improve enough: {in_snr:.2f} -> {out_snr:.2f} dB"


def test_enhancer_is_deterministic():
    """Two independent runs on the same input produce identical output
    (the C engine is deterministic)."""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(4 * FRAME_SIZE) * 0.1).astype(np.float32)
    with Enhancer() as enh:
        a = _enhance_signal(enh, x)
        enh.reset()
        b = _enhance_signal(enh, x)
    assert np.array_equal(a, b)
