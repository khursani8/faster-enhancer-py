"""ctypes binding to the C engine (``libfe`` shared library).

This is the implementation backend for :class:`faster_enhancer.Enhancer`. It
locates the compiled shared library shipped inside the package, declares the
four-function C ABI (``fe_init`` / ``fe_run`` / ``fe_reset`` / ``fe_free``),
and wraps it in a Pythonic, single-instance engine.

The C runtime owns a single global engine instance (see ``include/fe.h``), so
at most one :class:`Enhancer` may be live at a time; constructing a second one
without :meth:`close`-ing the first raises :class:`RuntimeError`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_float, c_int, c_void_p
from pathlib import Path

import numpy as np

# Public ABI constants (mirrors include/fe.h).
FRAME_SIZE = 320
SAMPLE_RATE = 48000

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = _PKG_DIR / "weights" / "fe.q8"

# Shared-library filename patterns per platform (OUTPUT_NAME is "fe").
_LIB_PATTERNS = {
    "linux": ["libfe.so", "libfe.so.*"],
    "darwin": ["libfe.dylib", "libfe.*.dylib"],
    "win32": ["fe.dll", "libfe.dll"],
}


def _platform_key() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "win32"
    return "linux"


def _find_library(search_path: str | os.PathLike | None = None) -> str:
    """Resolve the path to the bundled ``libfe`` shared library."""
    env = os.environ.get("FASTER_ENHANCER_LIB")
    if env and Path(env).exists():
        return env
    dirs = [_PKG_DIR]
    if search_path is not None:
        dirs.insert(0, Path(search_path))
    for d in dirs:
        for pat in _LIB_PATTERNS.get(_platform_key(), []):
            hits = sorted(d.glob(pat))
            if hits:
                return str(hits[0])
    raise OSError(
        "libfe shared library not found under the package directory "
        f"({_PKG_DIR}). The wheel may not have been built for this platform; "
        "reinstall, or set FASTER_ENHANCER_LIB to an explicit library path."
    )


class _Lib:
    """Lazy-loaded, prototype-decorated handle to libfe."""

    def __init__(self, path: str):
        self.path = path
        self.cdll = ctypes.CDLL(path)
        c = self.cdll
        # fe_init(const void* blob, int size) -> int. c_char_p accepts a Python
        # bytes object and passes a pointer to its stable internal buffer; the
        # blob must outlive the engine (kept on the Enhancer instance).
        c.fe_init.argtypes = [c_char_p, c_int]
        c.fe_init.restype = c_int
        c.fe_run.argtypes = [POINTER(c_float), POINTER(c_float)]
        c.fe_run.restype = None
        c.fe_reset.argtypes = []
        c.fe_reset.restype = None
        c.fe_free.argtypes = []
        c.fe_free.restype = None


_LIB: _Lib | None = None
# The C engine holds a single global instance; enforce singleton-ness here.
_LIVE_INSTANCES = 0


def _get_lib(search_path: str | os.PathLike | None = None) -> _Lib:
    global _LIB
    if _LIB is None or search_path is not None:
        _LIB = _Lib(_find_library(search_path))
    return _LIB


def load_default_weights() -> bytes:
    """Return the vendored production weights blob (``weights/fe.q8``)."""
    return DEFAULT_WEIGHTS_PATH.read_bytes()


class Enhancer:
    """Streaming FastEnhancer-Medium speech enhancer backed by the C engine.

    Parameters
    ----------
    weights_blob:
        Contents of an ``fe.q8`` blob. If ``None``, the vendored production
        blob is used.
    lib_path:
        Optional explicit path to a ``libfe`` shared library (else the bundled
        one is used). Mainly for development.

    Notes
    -----
    Single-threaded, single-instance (matches the C ABI). Process one
    ``FRAME_SIZE``-sample (320-sample, 6.67 ms @ 48 kHz) frame per call.
    """

    def __init__(
        self,
        weights_blob: bytes | None = None,
        *,
        lib_path: str | os.PathLike | None = None,
    ):
        global _LIVE_INSTANCES
        if _LIVE_INSTANCES != 0:
            raise RuntimeError(
                "Only one Enhancer may be live at a time (the C engine uses a "
                "single global instance). Call .close() on the existing one first."
            )
        if weights_blob is None:
            weights_blob = load_default_weights()
        if not isinstance(weights_blob, (bytes, bytearray)):
            raise TypeError("weights_blob must be bytes (the contents of an fe.q8 blob).")

        self._lib = _get_lib(lib_path)
        # The blob is referenced zero-copy by the engine until fe_free().
        self._blob = bytes(weights_blob)
        rc = self._lib.cdll.fe_init(self._blob, len(self._blob))
        if rc != 0:
            raise RuntimeError(
                f"fe_init returned {rc}: unsupported CPU or corrupt weights blob."
            )
        _LIVE_INSTANCES += 1
        self._closed = False

    # ------------------------------------------------------------------ #
    def run(self, audio_in) -> np.ndarray:
        """Enhance one ``FRAME_SIZE``-sample frame; returns ``float32[320]``.

        ``audio_in`` may be any 1-D array-like of length 320 (float32).
        """
        if self._closed:
            raise RuntimeError("Enhancer is closed.")
        x = np.ascontiguousarray(audio_in, dtype=np.float32)
        if x.shape != (FRAME_SIZE,):
            raise ValueError(f"expected {FRAME_SIZE} samples, got shape {x.shape}.")
        out = np.empty(FRAME_SIZE, dtype=np.float32)
        self._lib.cdll.fe_run(
            x.ctypes.data_as(POINTER(c_float)),
            out.ctypes.data_as(POINTER(c_float)),
        )
        return out

    def reset(self) -> None:
        """Clear streaming state (GRU hidden + STFT/iSTFT caches). Call between
        independent audio streams."""
        if not self._closed:
            self._lib.cdll.fe_reset()

    def close(self) -> None:
        """Release the C engine. Required before constructing another Enhancer."""
        global _LIVE_INSTANCES
        if not self._closed:
            try:
                self._lib.cdll.fe_free()
            finally:
                self._closed = True
                _LIVE_INSTANCES = max(0, _LIVE_INSTANCES - 1)

    def __enter__(self) -> "Enhancer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
