"""Method comparison — time the four transforms on real audio, side by side.

Times STFT vs PyWavelet CWT vs CWT (CPU) vs CWT (GPU) on the same chunks of a
real song, and records each method's *native* frequency-resolution character
alongside its cost. They all resample to the same chromatic output grid, so the
interesting difference is the underlying resolution model:

    STFT          linear  — one fixed Hz spacing everywhere (sample_rate/nperseg).
                            Too coarse to separate low notes (a bass semitone is
                            < 2 Hz), wasteful up high.
    CWT (all 3)   log     — constant-Q: one *semitone* everywhere, so it resolves
                            notes at every octave. This is the project's thesis.

Results go to ``assets/timing/timing_methods.csv``; the report's Overview table
reads the latest set. This benchmark is pure DSP — no renderer / window — so it
runs anywhere CuPy + the audio file are available.

Usage:
    python research/test_suite.py --methods
"""

import os

import numpy as np
import soundfile as sf

from subshader.config import CWTConfig
from subshader.dsp.stft import STFT
from subshader.dsp.pywavelet import PywaveletCWT
from subshader.dsp.cwt import CpuCWT, GpuCWT
from subshader.utils.gpu import gpu_available

from utilities import (
    AUDIO_BELTRAN, time_call, METHODS_CSV, append_method_rows, methods_table_md,
)
from utilities.timing_results import git_sha, _timestamp

CHUNK_SIZE = 16384
# PyWavelet's pywt.cwt is ~50 s PER CALL over 116 scales, so keep the chunk count
# tiny — its cost is deterministic, one or two samples give a representative mean.
# The other three methods are milliseconds, so a small N is fine for them too.
N_CHUNKS = 2
WARMUP = 1        # one untimed call first (mainly to plan the GPU cuFFT / pywt setup)


def _load_chunks(path, chunk_size, n_chunks):
    """Return (chunks, sample_rate, total_samples): n evenly-spaced mono windows."""
    data, sr = sf.read(path, dtype="float64", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    total = len(data)
    step = max(1, (total - chunk_size) // n_chunks)
    starts = [i * step for i in range(n_chunks) if i * step + chunk_size <= total]
    return [data[s:s + chunk_size].copy() for s in starts], float(sr), total


def _resolution(method_obj, sr, chunk_size, kind):
    """(native_res string, res_kind) for a method.

    STFT: uniform Hz spacing (sample_rate / nperseg) — linear.
    CWT family: a semitone everywhere — log / constant-Q. The absolute Hz gap is
    smallest at the lowest note, so we report that bottom gap to expose how STFT's
    fixed bin is too coarse there.
    """
    if kind == "linear":
        hz = sr / float(method_obj.nperseg)
        return f"{hz:.1f} Hz uniform", "linear"
    freqs = np.asarray(method_obj.freqs, dtype=float)
    semitone_pct = (freqs[1] / freqs[0] - 1.0) * 100.0 if len(freqs) > 1 else 0.0
    bottom_gap = (freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    return f"1 semitone (~{semitone_pct:.0f}%, {bottom_gap:.1f} Hz at {freqs[0]:.0f} Hz)", "log"


def _num_freqs(method_obj):
    """Output frequency-bin count for any backend."""
    if hasattr(method_obj, "num_freqs"):
        return int(method_obj.num_freqs)
    return int(len(method_obj.freqs))


def _time_method(obj, chunks):
    """Warm up, then time one .process() per chunk. Returns (mean_ms, std_ms)."""
    for _ in range(WARMUP):
        obj.process(chunks[0])
    times = [time_call(obj.process, c)[1] for c in chunks]
    arr = np.asarray(times, dtype=float)
    return float(arr.mean()), float(arr.std())


def run_methods(audio_path=AUDIO_BELTRAN, chunk_size=CHUNK_SIZE, n_chunks=N_CHUNKS):
    """Benchmark all available methods on the clip; append to the methods CSV."""
    print(f"\nSubShader Method Comparison — {n_chunks} chunks of "
          f"{os.path.basename(audio_path)} @ chunk {chunk_size}\n")

    chunks, sr, total = _load_chunks(audio_path, chunk_size, n_chunks)
    config = CWTConfig(file_path=audio_path, chunk_size=chunk_size,
                       sample_rate=sr, total_samples=total)

    # (label, backend, factory, resolution-kind). GPU last; skipped if unavailable.
    specs = [
        ("SciPy STFT",    "CPU", lambda: STFT(config),         "linear"),
        ("PyWavelet CWT", "CPU", lambda: PywaveletCWT(config), "log"),
        ("CWT",           "CPU", lambda: CpuCWT(config),       "log"),
    ]
    if gpu_available():
        specs.append(("CWT", "GPU", lambda: GpuCWT(config), "log"))
    else:
        print("  GPU unavailable — skipping GpuCWT.")

    ts, sha = _timestamp(), git_sha()
    rows = []
    freq_lo = freq_hi = 0.0  # chromatic output grid (same for all methods); set from a log method
    for label, backend, factory, kind in specs:
        obj = factory()
        mean_ms, std_ms = _time_method(obj, chunks)
        native_res, res_kind = _resolution(obj, sr, chunk_size, kind)
        nfreqs = _num_freqs(obj)
        if res_kind == "log" and freq_hi == 0.0:
            f = np.asarray(obj.freqs, dtype=float)
            freq_lo, freq_hi = float(f.min()), float(f.max())
        if hasattr(obj, "cleanup"):
            try:
                obj.cleanup()
            except Exception:
                pass
        rows.append({
            "timestamp": ts, "git_sha": sha, "method": label, "backend": backend,
            "chunk_size": chunk_size, "num_freqs": nfreqs,
            "sample_rate": round(sr, 2), "freq_lo": 0.0, "freq_hi": 0.0,
            "mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
            "native_res": native_res, "res_kind": res_kind,
        })
        print(f"  {label:<14} {backend:<3}  →  {mean_ms:8.2f} ms   "
              f"({nfreqs} bins · {native_res})")

    # Stamp the shared chromatic frequency range onto every row.
    for row in rows:
        row["freq_lo"] = round(freq_lo, 2)
        row["freq_hi"] = round(freq_hi, 2)

    append_method_rows(rows)
    # Refresh TIMING.md so the Overview's method table picks up these numbers.
    from timing_report import update_report
    update_report()
    print(f"\n  numbers  →  {METHODS_CSV}")
    print(f"  table     →  {os.path.join(os.path.dirname(METHODS_CSV), 'TIMING.md')}\n")
    print(methods_table_md())
    return rows


if __name__ == "__main__":
    run_methods()
