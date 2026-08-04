"""
Streaming VAD gate for faster-enhancer-py.

Wraps the Enhancer with Silero VAD post-processing.
Adds ~33ms latency (5 frames buffered for VAD decision).

Usage:
    from faster_enhancer import FRAME_SIZE, SAMPLE_RATE
    from faster_enhancer.vad_gate import VADGatedEnhancer

    with VADGatedEnhancer() as enh:
        for each 320-sample frame at 48kHz:
            out = enh.run(noisy_frame)  # enhanced + VAD-gated
"""
from __future__ import annotations

import numpy as np
import torch

from . import Enhancer, FRAME_SIZE, SAMPLE_RATE

VAD_SR = 16000
VAD_CHUNK_16K = 512  # Silero requires exactly 512 samples at 16kHz
VAD_CHUNK_48K = VAD_CHUNK_16K * SAMPLE_RATE // VAD_SR  # 1536 samples at 48kHz
# Number of enhancer frames per VAD call
FRAMES_PER_VAD = (VAD_CHUNK_48K + FRAME_SIZE - 1) // FRAME_SIZE  # 5


class VADGatedEnhancer:
    """Streaming enhancer with VAD-gated silence suppression.

    Pipeline: noisy → C engine → enhanced → VAD gate → output

    Latency: +33ms (5 frames buffered for one VAD decision).
    First 5 calls return zeros (buffer fill-up).

    Parameters
    ----------
    vad_threshold : float
        Speech probability above this = speech (default 0.5).
    attenuation_db : float
        Non-speech gain in dB (default -40 ≈ near-silent).
    smoothing : float
        EMA smoothing for gain transitions, 0=no smoothing, 1=frozen (default 0.8).
    weights_blob : bytes or None
        Custom enhancer weights (default: vendored fe.q8).
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        attenuation_db: float = -40.0,
        smoothing: float = 0.8,
        weights_blob: bytes | None = None,
    ):
        self._enhancer = Enhancer(weights_blob)
        self._vad_threshold = vad_threshold
        self._attenuation_gain = 10 ** (attenuation_db / 20.0)
        self._smoothing = smoothing

        # Load Silero VAD
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
        )
        self._vad = model.to('cpu')
        self._vad.eval()
        self._vad.reset_states()

        # Streaming buffers
        self._enh_buffer = np.zeros(0, dtype=np.float32)  # buffered enhanced 48kHz
        self._out_buffer = np.zeros(0, dtype=np.float32)   # gated output ready to send
        self._current_gain = 1.0  # smoothed gain

    def run(self, audio_in: np.ndarray) -> np.ndarray:
        """Process one frame (320 samples at 48kHz). Returns float32[320].

        Output is delayed by ~33ms (5 frames) for VAD buffering.
        """
        # 1. Enhance
        enhanced = self._enhancer.run(audio_in)

        # 2. Buffer
        self._enh_buffer = np.concatenate([self._enh_buffer, enhanced])

        # 3. When buffer has enough for VAD, process
        if len(self._enh_buffer) >= VAD_CHUNK_48K:
            chunk = self._enh_buffer[:VAD_CHUNK_48K]
            self._enh_buffer = self._enh_buffer[VAD_CHUNK_48K:]

            # Downsample 48k → 16k for VAD
            chunk_16k = chunk[::3].astype(np.float32)[:VAD_CHUNK_16K]

            # Run VAD
            with torch.no_grad():
                prob = self._vad(
                    torch.from_numpy(chunk_16k).float(), VAD_SR
                ).item()

            # Compute target gain
            if prob > self._vad_threshold:
                target = 1.0
            else:
                target = self._attenuation_gain

            # Smooth gain transition
            self._current_gain = self._smoothing * self._current_gain + \
                                 (1 - self._smoothing) * target

            # Apply gain and add to output buffer
            gated = chunk * self._current_gain
            self._out_buffer = np.concatenate([self._out_buffer, gated])

        # 4. Return one frame from output buffer
        result = np.zeros(FRAME_SIZE, dtype=np.float32)
        if len(self._out_buffer) >= FRAME_SIZE:
            result = self._out_buffer[:FRAME_SIZE]
            self._out_buffer = self._out_buffer[FRAME_SIZE:]
        elif len(self._out_buffer) > 0:
            result[:len(self._out_buffer)] = self._out_buffer
            self._out_buffer = np.zeros(0, dtype=np.float32)

        return result

    def reset(self):
        """Reset all streaming state."""
        self._enhancer.reset()
        self._vad.reset_states()
        self._enh_buffer = np.zeros(0, dtype=np.float32)
        self._out_buffer = np.zeros(0, dtype=np.float32)
        self._current_gain = 1.0

    def close(self):
        self._enhancer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
