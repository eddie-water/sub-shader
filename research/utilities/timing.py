"""Timing utilities for the benchmark suite."""

import time
import numpy as np


def time_call(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs), return (result, elapsed_ms).

    Measures wall-clock time using perf_counter and returns the function
    result along with elapsed time in milliseconds.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


class TimingAccumulator:
    """
    Pre-allocated per-method timing for a frame loop.

    Usage:
        acc = TimingAccumulator(n_frames, ["STFT", "PyWavelet", "SubShader"])
        for i in range(n_frames):
            out, acc["STFT"][i]      = time_call(stft_fn, chunk)
            out, acc["PyWavelet"][i] = time_call(pywt_fn, chunk)
            out, acc["SubShader"][i] = time_call(npwt_fn, chunk)
            acc.advance()
        acc.trim()
        stats = acc.compute_stats()  # {"STFT": {"avg_ms":…, "min_ms":…, "max_ms":…}, …}
    """

    def __init__(self, n_frames, methods):
        """Initialize accumulator with pre-allocated arrays for each method."""
        self.n_frames = n_frames
        self.methods = methods
        self.arrays = {method: np.empty(n_frames) for method in methods}
        self.current_idx = 0

    def __getitem__(self, method):
        """Return the timing array for a given method."""
        if method not in self.methods:
            raise KeyError(f"Unknown method: {method!r}")
        return self.arrays[method]

    def advance(self):
        """Increment the current frame index."""
        self.current_idx += 1

    def trim(self):
        """Trim all timing arrays to the actual number of frames processed."""
        for method in self.methods:
            self.arrays[method] = self.arrays[method][:self.current_idx]

    def compute_stats(self):
        """Compute avg/min/max statistics for all methods.

        Returns:
            dict mapping method name to {"avg_ms": ..., "min_ms": ..., "max_ms": ...}
        """
        stats = {}
        for method in self.methods:
            times = self.arrays[method]
            stats[method] = {
                "avg_ms": float(np.mean(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
            }
        return stats
