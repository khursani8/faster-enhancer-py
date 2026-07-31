#!/usr/bin/env python3
"""Enhance a raw 48 kHz mono float32 file with faster-enhancer-py.

Input/output format: raw IEEE float32 little-endian, mono, 48 kHz
(``ffmpeg -ar 48000 -ac 1 -f f32le``). The fixed 704-sample STFT alignment
delay means the first ~0.9 ms of output is transient fill.

Usage:
    python enhance_file.py noisy.f32 --out enhanced.f32
    python enhance_file.py noisy.f32 --out enhanced.f32 --weights custom.q8
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from faster_enhancer import FRAME_SIZE, Enhancer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="raw float32 mono 48 kHz input file")
    ap.add_argument("--out", required=True, help="raw float32 output file")
    ap.add_argument("--weights", default=None,
                    help="path to an fe.q8 blob (default: vendored weights)")
    args = ap.parse_args()

    blob = open(args.weights, "rb").read() if args.weights else None
    audio = np.fromfile(args.input, dtype=np.float32)
    n_full = (audio.size // FRAME_SIZE) * FRAME_SIZE
    if n_full == 0:
        print("input shorter than one frame (320 samples); nothing to do.", file=sys.stderr)
        return 1
    audio = audio[:n_full]

    n_frames = audio.size // FRAME_SIZE
    out = np.empty_like(audio)
    with Enhancer(blob) as enh:
        for i in range(n_frames):
            s = i * FRAME_SIZE
            out[s:s + FRAME_SIZE] = enh.run(audio[s:s + FRAME_SIZE])
    out.tofile(args.out)
    print(f"processed {n_frames} frames ({audio.size} samples) -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
