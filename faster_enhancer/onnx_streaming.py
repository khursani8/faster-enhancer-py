"""
Production streaming speech enhancement for contact centre.

Uses ONNX Runtime with FastEnhancer-S model + VAD gate.
Runs real-time on CPU (RTF=0.034).

Usage:
    python onnx_streaming.py --input call.wav --output enhanced.wav
    python onnx_streaming.py --input-dir calls/ --output-dir enhanced/
"""
import argparse
import os
import sys
import time
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
from tqdm import tqdm

sys.path.insert(0, ".")
from vad_gate import vad_gate


# Model constants
ONNX_PATH = "models/fastenhancer_s_ep20.onnx"
HOP_SIZE = 256
SR = 16000
CHUNK_SEC = 2.0
CHUNK_SIZE = int(CHUNK_SEC * SR)


class StreamingEnhancer:
    """ONNX-based streaming speech enhancer with VAD gate.

    Pipeline: noisy → ONNX model → spectral gate → VAD gate → normalize → output
    """

    def __init__(self, onnx_path=ONNX_PATH, use_vad=True, use_spectral=True,
                 target_db=-14.0):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.use_vad = use_vad
        self.use_spectral = use_spectral
        self.target_db = target_db

        # Load VAD if needed
        if use_vad:
            import torch
            self._torch = torch
            self._load_vad()

        print(f"Loaded: {onnx_path}")
        print(f"VAD gate: {'ON' if use_vad else 'OFF'}")
        print(f"Spectral gate: {'ON' if use_spectral else 'OFF'}")

    def _load_vad(self):
        model, utils = self._torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
        )
        self._vad = model.to('cpu')
        self._vad.eval()
        self._get_timestamps = utils[0]

    def _spectral_gate(self, wav_np):
        """Lightweight spectral noise gate."""
        import torch
        from inference import spectral_gate as _sg
        wav_t = torch.from_numpy(wav_np).float()
        return _sg(wav_t, hop=HOP_SIZE).numpy()

    def _vad_gate(self, wav_np):
        """VAD-gated silence suppression."""
        import torch
        wav_t = torch.from_numpy(wav_np).float()
        gated = vad_gate(wav_t, sr=SR, attenuation_db=-40.0)
        return gated.numpy()

    def _normalize(self, wav_np):
        """Compress + normalize for consistent loudness."""
        import torch
        from inference import normalize_loudness
        wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
        return normalize_loudness(wav_t, target_db=self.target_db).squeeze().numpy()

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Enhance one chunk of audio (any length divisible by hop_size)."""
        # Pad if needed
        if len(audio) % HOP_SIZE != 0:
            audio = np.pad(audio, (0, HOP_SIZE - len(audio) % HOP_SIZE))

        # 1. ONNX model
        inp = audio[np.newaxis, :].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: inp})
        enhanced = outputs[0].squeeze()

        # 2. Spectral gate
        if self.use_spectral:
            enhanced = self._spectral_gate(enhanced)

        # 3. VAD gate
        if self.use_vad:
            enhanced = self._vad_gate(enhanced)

        # 4. Normalize loudness
        enhanced = self._normalize(enhanced)

        return enhanced[:len(audio)]

    def enhance_file(self, input_path: str, output_path: str):
        """Enhance a full audio file."""
        audio, sr = librosa.load(input_path, sr=SR, mono=True)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        # Process in chunks
        output = np.zeros(len(audio), dtype=np.float32)
        n_chunks = (len(audio) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for i in range(n_chunks):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(audio))
            chunk = audio[start:end]
            if len(chunk) < HOP_SIZE:
                break
            enhanced = self.enhance_chunk(chunk)
            output[start:end] = enhanced[:end-start]

        sf.write(output_path, output, SR)
        return output


def main():
    parser = argparse.ArgumentParser(description="Streaming speech enhancement")
    parser.add_argument("--input", help="Input audio file")
    parser.add_argument("--output", default="enhanced_output.wav")
    parser.add_argument("--input-dir", help="Batch process directory")
    parser.add_argument("--output-dir", help="Batch output directory")
    parser.add_argument("--onnx", default=ONNX_PATH)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD gate")
    parser.add_argument("--no-spectral", action="store_true", help="Disable spectral gate")
    parser.add_argument("--target-db", type=float, default=-14.0)
    args = parser.parse_args()

    enhancer = StreamingEnhancer(
        onnx_path=args.onnx,
        use_vad=not args.no_vad,
        use_spectral=not args.no_spectral,
        target_db=args.target_db,
    )

    if args.input:
        print(f"Input: {args.input}")
        t0 = time.perf_counter()
        enhancer.enhance_file(args.input, args.output)
        elapsed = time.perf_counter() - t0
        info = sf.info(args.input)
        print(f"Output: {args.output} ({elapsed:.2f}s for {info.duration:.1f}s audio, RTF={elapsed/info.duration:.3f})")

    elif args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = {".wav", ".flac", ".mp3"}
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(args.input_dir)
                 for f in fs if os.path.splitext(f)[1].lower() in exts]
        print(f"Processing {len(files)} files...")

        for fpath in tqdm(files):
            rel = os.path.relpath(fpath, args.input_dir)
            out = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".wav")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            enhancer.enhance_file(fpath, out)

        print(f"Done. {len(files)} files → {args.output_dir}")


if __name__ == "__main__":
    main()
