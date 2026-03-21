"""GPU detection utility for SubShader."""

from subshader.utils.logging import get_logger

log = get_logger(__name__)


def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible via CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False
