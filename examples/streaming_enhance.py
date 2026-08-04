"""
Streaming ONNX speech enhancement for contact centre telephony.

Real-time on CPU (RTF=0.034 without VAD, ~0.49 with VAD).
Optional VAD gate kills residual noise in silence regions.

Usage:
  # Single file (no VAD — fastest)
  python -m faster_enhancer.onnx_streaming --input call.wav --output clean.wav

  # With VAD gate (kills silence noise)
  python -m faster_enhancer.onnx_streaming --input call.wav --output clean.wav --vad

  # Batch processing
  python -m faster_enhancer.onnx_streaming --input-dir calls/ --output-dir enhanced/ --vad

  # Python API for streaming
  from faster_enhancer.onnx_streaming import StreamingEnhancer
  enh = StreamingEnhancer(vad=True)
  enhanced = enh.enhance_file("call.wav", "clean.wav")
  # Or frame-by-frame:
  for frame in audio_chunks:
      out = enh.enhance_frame(frame)  # 256 samples in → 256 out
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import librosa
import soundfile as sf
import onnxruntime as ort
from tqdm import tqdm


class StreamingEnhancer:
    """ONNX-based streaming speech enhancer.

    Uses the FastEnhancer ONNX model (256 samples/frame, stateful).
    Optional VAD gate (Silero) suppresses residual noise in silence.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file.
    vad : bool
        Enable Silero VAD gate (kills noise in non-speech regions).
    spectral_gate : bool
        Enable spectral noise gate (cleans residual hiss).
    normalize_db : float
        Target output loudness in dB (-14 = comfortable listening).
    """

    def __init__(
        self,
        onnx_path: str = None,
        vad: bool = False,
        spectral_gate: bool = False,
        normalize_db: float = -14.0,
    ):
        # Find ONNX model
        if onnx_path is None:
            script_dir = Path(__file__).resolve().parent
            pkg_dir = script_dir.parent / "faster_enhancer"
            candidates = [
                pkg_dir / "weights" / "fastenhancer.onnx",
                script_dir / "fastenhancer.onnx",
                Path("fastenhancer.onnx"),
                Path("models/fastenhancer_s_ep20.onnx"),
            ]
            for c in candidates:
                if c.exists():
                    onnx_path = str(c)
                    break
            else:
                raise FileNotFoundError(
                    "ONNX model not found. Pass onnx_path explicitly."
                )

        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.frame_size = 256  # samples per frame (hop_size)
        self.normalize_db = normalize_db

        # Map streaming cache: output names → input names
        inputs = [i for i in self.session.get_inputs() if i.name != 'wav_in']
        outputs = [o for o in self.session.get_outputs() if o.name != 'wav_out']
        self._cache_map = dict(zip([o.name for o in outputs], [i.name for i in inputs]))
        self._cache_shapes = {i.name: i.shape for i in inputs}

        # VAD gate
        self._vad_enabled = vad
        self._vad_model = None
        if vad:
            self._load_vad()

        # Spectral gate
        self._sg_enabled = spectral_gate

        print(f"StreamingEnhancer loaded: {onnx_path}")
        print(f"  VAD gate: {'ON' if vad else 'OFF'}")
        print(f"  Spectral gate: {'ON' if spectral_gate else 'OFF'}")
        print(f"  Normalize: {normalize_db} dB")

    def _load_vad(self):
        """Load Silero VAD for silence detection."""
        import torch
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
        )
        self._torch = torch
        self._vad_model = model.to('cpu')
        self._vad_model.eval()
        self._get_timestamps = utils[0]
        self._vad_gain = 1.0  # smoothed gain
        self._vad_smoothing = 0.8

    def _init_cache(self) -> dict:
        """Initialize streaming cache (GRU hidden states + STFT buffers)."""
        cache = {}
        for name, shape in self._cache_shapes.items():
            dims = [d if isinstance(d, int) else 1 for d in shape]
            cache[name] = np.zeros(dims, dtype=np.float32)
        return cache

    def enhance_frame(self, frame: np.ndarray, cache: dict = None) -> tuple:
        """Enhance one streaming frame (256 samples at 16kHz).

        Returns (enhanced_frame, updated_cache).
        For true streaming: pass cache from previous call.
        """
        if cache is None:
            cache = self._init_cache()

        if len(frame) != self.frame_size:
            frame = frame[:self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

        feed = {'wav_in': frame[np.newaxis, :].astype(np.float32)}
        feed.update(cache)

        outputs = self.session.run(None, feed)

        enhanced = outputs[0].squeeze()

        # Update cache
        new_cache = {}
        for j, out in enumerate(self.session.get_outputs()):
            if out.name != 'wav_out' and out.name in self._cache_map:
                new_cache[self._cache_map[out.name]] = outputs[j]

        return enhanced, new_cache

    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance a full audio array (any length, 16kHz mono)."""
        # Pad to frame_size multiple
        pad = (-len(audio)) % self.frame_size
        if pad:
            audio = np.pad(audio, (0, pad))

        cache = self._init_cache()
        enhanced = np.zeros_like(audio)

        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]
            out, cache = self.enhance_frame(frame, cache)
            enhanced[i:i + self.frame_size] = out

        # Post-processing
        if self._sg_enabled:
            enhanced = self._apply_spectral_gate(enhanced)

        if self._vad_enabled:
            enhanced = self._apply_vad_gate(enhanced)

        enhanced = self._normalize(enhanced)

        return enhanced[:len(audio) - pad] if pad else enhanced

    def _apply_spectral_gate(self, wav: np.ndarray) -> np.ndarray:
        """Lightweight spectral noise gate."""
        import torch
        n_fft = 512
        hop = 256
        window = torch.hann_window(n_fft)
        stft = torch.stft(
            torch.from_numpy(wav).float(), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True
        )
        mag = stft.abs()
        frame_energy = mag.mean(dim=0)
        num_noise = max(1, int(len(frame_energy) * 0.1))
        noise_idx = frame_energy.argsort()[:num_noise]
        noise_floor = mag[:, noise_idx].mean(dim=1)
        sig_power = mag ** 2
        noise_power = noise_floor.unsqueeze(-1) ** 2
        gain = ((sig_power - 2.0 * noise_power) / (sig_power + 1e-8)).clamp(0, 1)
        gain = gain.sqrt()
        cleaned = stft * gain
        result = torch.istft(cleaned, n_fft=n_fft, hop_length=hop, window=window)
        return result.numpy()[:len(wav)]

    def _apply_vad_gate(self, wav: np.ndarray) -> np.ndarray:
        """VAD-gated silence suppression using Silero VAD."""
        audio_t = self._torch.from_numpy(wav).float()
        timestamps = self._get_timestamps(
            audio_t.numpy(), self._vad_model, threshold=0.5,
        )

        mask = np.zeros(len(wav))
        for ts in timestamps:
            mask[ts['start']:min(ts['end'], len(wav))] = 1.0

        # Dilate speech regions by 30ms (avoid clipping speech)
        dilate = int(16000 * 0.03)
        dilated = mask.copy()
        speech_idx = np.where(mask > 0)[0]
        for idx in speech_idx:
            start = max(0, idx - dilate)
            end = min(len(wav), idx + dilate)
            dilated[start:end] = 1.0

        # Apply attenuation to non-speech
        attenuation = 10 ** (-40 / 20)  # -40 dB
        gain = dilated * (1.0 - attenuation) + attenuation
        return wav * gain

    def _normalize(self, wav: np.ndarray) -> np.ndarray:
        """Compress + normalize to target dB."""
        rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
        target = 10 ** (self.normalize_db / 20)
        wav = wav * (target / (rms + 1e-8))
        # Gentle compression on peaks
        abs_w = np.abs(wav)
        excess = np.maximum(abs_w - 0.5, 0)
        wav = wav - np.sign(wav) * excess * 0.5
        return np.clip(wav, -0.95, 0.95).astype(np.float32)

    def enhance_file(self, input_path: str, output_path: str) -> float:
        """Enhance an audio file. Returns RTF (lower = faster)."""
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        t0 = time.perf_counter()
        enhanced = self.enhance_audio(audio)
        elapsed = time.perf_counter() - t0

        sf.write(output_path, enhanced, 16000)
        duration = len(audio) / 16000
        rtf = elapsed / duration
        return rtf


def main():
    parser = argparse.ArgumentParser(
        description="Streaming ONNX speech enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Input audio file")
    parser.add_argument("--output", "-o", default="enhanced.wav", help="Output file")
    parser.add_argument("--input-dir", help="Batch process directory")
    parser.add_argument("--output-dir", help="Batch output directory")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model")
    parser.add_argument("--vad", action="store_true", help="Enable VAD gate")
    parser.add_argument("--spectral-gate", action="store_true", help="Enable spectral gate")
    parser.add_argument("--normalize-db", type=float, default=-14.0)
    args = parser.parse_args()

    enhancer = StreamingEnhancer(
        onnx_path=args.onnx,
        vad=args.vad,
        spectral_gate=args.spectral_gate,
        normalize_db=args.normalize_db,
    )

    if args.input:
        rtf = enhancer.enhance_file(args.input, args.output)
        info = sf.info(args.input)
        print(f"\n{args.input} → {args.output}")
        print(f"  Duration: {info.duration:.1f}s")
        print(f"  RTF: {rtf:.3f} ({'real-time ✓' if rtf < 1.0 else 'NOT real-time'})")

    elif args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = {".wav", ".flac", ".mp3", ".ogg"}
        files = [
            os.path.join(dp, f)
            for dp, _, fs in os.walk(args.input_dir)
            for f in fs
            if os.path.splitext(f)[1].lower() in exts
        ]
        print(f"Processing {len(files)} files...")
        for fpath in tqdm(files):
            rel = os.path.relpath(fpath, args.input_dir)
            out = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".wav")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            enhancer.enhance_file(fpath, out)
        print(f"Done. {len(files)} files → {args.output_dir}")

    else:
        parser.error("Provide --input or --input-dir")


if __name__ == "__main__":
    main()
