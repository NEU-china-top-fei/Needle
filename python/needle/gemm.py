"""Explicit GEMM precision/implementation policy for the CUDA backend.

CPU operations retain their normal FP32 path. Tensor Core modes round inputs
to FP16, accumulate in FP32 and return FP32; they are not strict FP32 GEMM.
Use the context around both forward and backward if testing training workloads.
"""
from contextlib import contextmanager
from contextvars import ContextVar

MODES = ("baseline", "simt", "wmma", "tensorcore")
_mode = ContextVar("needle_gemm_mode", default="baseline")


def get_gemm_mode():
    return _mode.get()


@contextmanager
def gemm_mode(mode):
    """Select baseline, FP32 SIMT, synchronous WMMA, or pipelined Tensor Core.

    Example: ``with ndl.gemm_mode('tensorcore'): y = model(x)``.
    Selection is context-local and restored even if execution raises an error.
    No FP32-to-FP16 conversion is performed unless explicitly selected.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown GEMM mode {mode!r}; expected one of {MODES}")
    token = _mode.set(mode)
    try:
        yield
    finally:
        _mode.reset(token)
