#!/usr/bin/env python3
"""Denoise demo: load a wav, mix in Gaussian noise at a target SNR, enhance,
and report how much of the original signal is recovered.

The engine runs at 48 kHz only, so the input is resampled to 48 kHz first.
Metrics are sample-aligned: the C engine has a fixed 704-sample (14.67 ms)
streaming alignment delay, so ``enhanced[n]`` is compared against
``clean[n-704]``. Output SNR vs input SNR shows the enhancement gain.

Usage:
    python denoise_demo.py anwar.wav --snr 0 --out-dir out
    python denoise_demo.py anwar.wav --snr 5 --no-noise  # passthrough (clean in)
Outputs out/{clean,noisy,enhanced}.wav at 48 kHz PCM_16, plus a metrics line.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 48000
FRAME = 320
ALIGN_DELAY = 704  # C engine STFT alignment delay (samples, 48 kHz)


def load_clean_48k(path: str) -> np.ndarray:
    """Load mono audio and resample to 48 kHz float32 in [-1, 1]."""
    import librosa
    y, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def add_noise_at_snr(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Return clean + scaled white Gaussian noise at the target SNR (dB)."""
    sig_p = float(np.mean(clean ** 2))
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    noise_p = float(np.mean(noise ** 2))
    scale = np.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10.0)))
    return clean + (noise * np.float32(scale))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """SNR of `test` against `reference` (both same length, aligned)."""
    p_ref = float(np.sum(reference ** 2))
    p_err = float(np.sum((reference - test) ** 2))
    return 10.0 * np.log10(p_ref / (p_err + 1e-20))


def enhance(noisy: np.ndarray, enh) -> np.ndarray:
    n = (noisy.shape[0] // FRAME) * FRAME
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, FRAME):
        out[i:i + FRAME] = enh.run(noisy[i:i + FRAME])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input wav (any sample rate, mono)")
    ap.add_argument("--snr", type=float, default=0.0, help="input SNR in dB (default 0)")
    ap.add_argument("--out-dir", default="out", help="where to write wav outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true",
                    help="pass clean audio through the enhancer (no noise added)")
    args = ap.parse_args()

    from faster_enhancer import Enhancer
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = load_clean_48k(args.input)
    rng = np.random.default_rng(args.seed)
    noisy = clean.copy() if args.no_noise else add_noise_at_snr(clean, args.snr, rng)

    with Enhancer() as enh:
        enhanced = enhance(noisy, enh)

    # Align by the 704-sample streaming delay: enhanced is the delayed stream,
    # so enhanced[n] estimates clean[n-704]. Drop the first `d` samples of the
    # enhanced output and compare against the un-delayed clean/noisy prefix.
    d = ALIGN_DELAY
    L = min(enhanced.shape[0] - d, clean.shape[0])
    enhanced_a = enhanced[d:d + L]
    clean_a = clean[:L]
    noisy_a = noisy[:L]

    input_snr = snr_db(clean_a, noisy_a)
    output_snr = snr_db(clean_a, enhanced_a)
    # Also report how much of the *noise* was removed (noisy->enhanced vs clean).
    print("=" * 64, file=sys.stderr)
    print(f"input  : {args.input}", file=sys.stderr)
    print(f"frames : {enhanced.shape[0] // FRAME} "
          f"({enhanced.shape[0] / SAMPLE_RATE:.2f}s @ 48 kHz)", file=sys.stderr)
    print(f"input SNR  (noisy  vs clean) : {input_snr:6.2f} dB", file=sys.stderr)
    print(f"output SNR (enhanced vs clean): {output_snr:6.2f} dB", file=sys.stderr)
    print(f"SNR improvement              : {output_snr - input_snr:+6.2f} dB", file=sys.stderr)
    print("=" * 64, file=sys.stderr)

    def write(name, x):
        x16 = np.clip(x, -1.0, 1.0)
        sf.write(out_dir / name, x16, SAMPLE_RATE, subtype="PCM_16")
    write("clean.wav", clean[:enhanced.shape[0]])
    write("noisy.wav", noisy[:enhanced.shape[0]])
    write("enhanced.wav", enhanced)
    print(f"wrote clean.wav, noisy.wav, enhanced.wav into {out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
