"""Research-only GPU transfer profiling for the CWT.

``GpuCWT.transform`` runs as one ``@timed`` blob. Because CuPy is asynchronous,
the individual GPU legs (upload, multiply, ifft, download) can't be measured by a
plain stopwatch — work is only *queued* until something blocks (the final
``cp.asnumpy`` download). The blob *total* is honest, but the per-leg split is not.

``InstrumentedGpuCWT`` subclasses the production ``GpuCWT`` and overrides
``transform`` to run the **identical** computation with a device sync after each
leg, so each leg's time is real. This lives in ``research/`` only — ``src/`` is
untouched and carries zero sync overhead in production.

Legs recorded per call in ``self.leg_times_ms``:
    fft_cpu   — CPU FFT of the input (numpy)
    upload    — host→device transfer of the FFT'd input   (transfer bottleneck)
    multiply  — elementwise kernel multiply on GPU
    ifft      — inverse FFT on GPU (+ trim)
    download  — device→host transfer of the trimmed result (transfer bottleneck)
"""

import time

import numpy as np
from numpy.fft import fft

from subshader.dsp.cwt import GpuCWT
from subshader.utils.timing import timed

try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    _CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    _CUPY = False


def _sync():
    """Block until all queued GPU work has finished."""
    cp.cuda.runtime.deviceSynchronize()


class InstrumentedGpuCWT(GpuCWT):
    """GpuCWT with per-leg GPU timing (syncs between legs). Research use only."""

    _is_instrumented = True

    @timed
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Identical to GpuCWT.transform, but times each leg with a device sync."""
        legs = {}
        clock = time.perf_counter

        t0 = clock()
        input_f = fft(data, self.max_conv_n)
        legs["fft_cpu"] = (clock() - t0) * 1000.0

        t0 = clock()
        input_f_gpu = cp.asarray(input_f, dtype=cp.complex64, order="C")
        _sync()
        legs["upload"] = (clock() - t0) * 1000.0

        t0 = clock()
        conv_f_gpu = input_f_gpu * self.kernel_f_bank_gpu
        _sync()
        legs["multiply"] = (clock() - t0) * 1000.0

        t0 = clock()
        conv_tf_gpu = cp_fft.ifft(conv_f_gpu, axis=1)
        conv_tf_trimmed = conv_tf_gpu[:, : self.input_n]
        _sync()
        legs["ifft"] = (clock() - t0) * 1000.0

        t0 = clock()
        out = cp.asnumpy(conv_tf_trimmed)
        legs["download"] = (clock() - t0) * 1000.0

        self.leg_times_ms = legs
        return out


# Stage labels emitted when transfer profiling is active (order = data flow).
GPU_LEG_LABELS = ["fft_cpu", "upload", "multiply", "ifft", "download"]
