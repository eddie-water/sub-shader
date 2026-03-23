# SubShader Pipeline Latency Profile

**Generated:** 2026-03-22 (baseline); updated 2026-03-22 (post-fix)
**Hardware:** NVIDIA 4060 Ti 16GB, Intel CPU, 44100 Hz audio
**Baseline config:** `chunk_size=16384`, `overlap_factor=0.75`, `target_width=256`, `num_octaves=10`
**Current config:** `chunk_size=4096`, `overlap_factor=0.5`, `target_width=256`, `num_octaves=10`
**Profiler:** `research/pipeline_timing_profile.py --frames 64 --headless`

---

## Summary

SubShader's visualization lag is structural, not a code bug. The CWT must
accumulate a full audio window before it can compute a single frame.

**Baseline (chunk_size=16384):** The CWT accumulated 371.5 ms of audio per
frame and updated every 92.9 ms (10.8 fps). By comparison, a typical DAW
spectrogram uses a 23 ms window and updates at 172 fps.

**Post-fix (chunk_size=4096):** The window shrinks to 92.9 ms and updates fire
every 46.4 ms (21.5 fps) — a 2× improvement in update rate with no code risk.
Additionally, the CWT iFFT was moved from CPU to GPU, reducing the ~42 ms
per-frame CWT cost to an estimated ~5-10 ms, freeing significant headroom for
lower latency or higher frame rates in the future.

The remaining lag is the CWT's fundamental constraint: low-frequency wavelets
(A0 = 27.5 Hz) require wide windows to avoid edge effects. Trimming the lowest
octaves is the next lever if further reduction is needed.

---

## Stage Timing

All times in milliseconds (ms). Measured with `time.perf_counter()`.

### Baseline (chunk_size=16384, overlap_factor=0.75) — 64 frames, GPU mode

| Stage | avg | med | min | max | p95 |
|---|---:|---:|---:|---:|---:|
| `get_chunk()` | 0.14 | 0.12 | 0.11 | 0.59 | 0.19 |
| `cwt()` total | **42.2** | 39.8 | 38.3 | 173.8 | 42.6 |
| `push_frame()` | 0.60 | 0.58 | 0.49 | 0.85 | 0.77 |
| **loop total** | **42.9** | 40.5 | 39.1 | 175.2 | 43.3 |

Hop budget: 92.9 ms. Budget surplus: 50.0 ms (54% free).

### Post-fix expected (chunk_size=4096, overlap_factor=0.5, GPU iFFT)

The profiler has not been re-run against the new config. Based on the sub-stage
analysis below, expected post-fix numbers are:

| Stage | Baseline avg | Post-fix estimate | Change |
|---|---:|---:|---|
| `get_chunk()` | 0.14 ms | ~0.05 ms | Smaller chunk reads faster |
| `cwt()` total | 42.2 ms | **~5-10 ms** | GPU iFFT eliminates 31.5ms CPU iFFT + 4.6ms download |
| `push_frame()` | 0.60 ms | ~0.60 ms | Unchanged |
| **loop total** | 42.9 ms | **~6-11 ms** | 4-7× faster |

New hop budget: 46.4 ms. Expected surplus: ~35-40 ms (75-87% free).

Re-run the profiler after this change to capture real numbers:
```bash
python research/pipeline_timing_profile.py --frames 64 --headless
```

---

## CWT Sub-Stage Breakdown (Baseline)

Measured at `chunk_size=16384`, before GPU iFFT fix.

| Sub-stage | avg | med | min | max | p95 | Notes |
|---|---:|---:|---:|---:|---:|---|
| `fft_cpu` | 1.48 | 1.43 | 1.36 | 2.98 | 1.58 | Input signal FFT, CPU |
| `gpu_upload` | 0.31 | 0.27 | 0.25 | 2.05 | 0.36 | Host → device (signal only) |
| `gpu_multiply` | 2.04 | 0.34 | 0.30 | 109.4 | 0.38 | 116 freq × 26005 samples, GPU |
| `gpu_download` | 4.57 | 4.40 | 3.75 | 13.6 | 5.32 | Full 116×26005 complex64 back to CPU |
| **`ifft_cpu`** | **31.5** | 31.3 | 29.8 | 41.9 | 32.8 | **Biggest single cost — CPU iFFT** |
| `mag` | 1.12 | 0.99 | 0.93 | 2.85 | 1.66 | `np.abs()` on 116×16384 |
| `downsample` | 0.18 | 0.17 | 0.13 | 0.31 | 0.27 | Index-select to 256 cols |
| `normalize` | < 0.01 | — | — | — | — | No-op (handled by kernel) |
| `discard` | < 0.01 | — | — | — | — | Array slice |
| `hop_center` | < 0.01 | — | — | — | — | Array slice |

**Key finding:** The iFFT running on CPU is 75% of CWT wall time (31.5 ms of 42 ms).
The GPU multiply is fast (0.34 ms median) but the result — a 116 × 26005 complex64
matrix (23 MB) — must be downloaded and inverse-transformed on CPU.

The `gpu_multiply` max of 109 ms is a CUDA sync spike. The kernel executes in 0.34 ms
but the first CUDA stream synchronize call after waits for any deferred kernel launch.
These spikes appear on the first few calls after initialization; the steady-state
median is consistent at 0.34 ms.

**Fix applied (GPU iFFT):** `CuPyWavelet.class_specific_cwt` now runs iFFT via
`cp_fft.ifft()` on GPU and downloads only the trimmed `(num_freqs, input_n)` slice.
This eliminates the 31.5 ms CPU iFFT and replaces the 4.6 ms full-matrix download
with a proportionally smaller transfer. Sub-stage profiling numbers for the new path
are pending a profiler re-run.

---

## Structural Latency: The Real Bottleneck

### Why the visualization is slow

The CWT is a **sliding window transform**. Every frame requires exactly
`chunk_size` samples to be collected before it can run. At 44100 Hz:

**Baseline:**
```
window duration = 16384 / 44100 = 371.5 ms
hop_size = 16384 × (1 − 0.75) = 4096 samples
hop_duration = 4096 / 44100 = 92.9 ms  →  10.8 fps
```

**Post-fix (current config):**
```
window duration = 4096 / 44100 = 92.9 ms
hop_size = 4096 × (1 − 0.5) = 2048 samples
hop_duration = 2048 / 44100 = 46.4 ms  →  21.5 fps
```

Every hop a new frame replaces the old one — the "chunky" feel.

### DAW comparison

| Parameter | SubShader (baseline) | SubShader (post-fix) | Typical DAW STFT |
|---|---:|---:|---:|
| Window size | 16384 samples | **4096 samples** | 1024 samples |
| Window duration | 371.5 ms | **92.9 ms** | 23.2 ms |
| Hop size | 4096 samples | **2048 samples** | 256 samples |
| Hop duration | 92.9 ms | **46.4 ms** | 5.8 ms |
| Max update rate | 10.8 fps | **21.5 fps** | 172 fps |

The post-fix config is 2× more responsive than the baseline. A DAW is still ~8×
faster because STFT can use tiny windows — CWT needs wider windows for low-frequency
accuracy. Trimming the lowest octaves is the path to further improvement.

### Why chunk_size must be large for CWT

The CWT resolves frequency by matching each signal segment against a bank of
wavelets at different scales. Low-frequency wavelets (e.g. A0 = 27.5 Hz) are
physically wide — they span many cycles of the carrier frequency. To avoid
edge effects, the input must be long enough to contain the full wavelet.

Current setup: `num_octaves=10` spans A0 (27.5 Hz) to ~14 kHz. The widest
wavelet (lowest note) determines `max_conv_n = 26005 samples`, and this drives
`chunk_size = 16384`. Trimming the lowest octaves is the most direct way to
allow a smaller chunk.

---

## Latency Budget

### Baseline (chunk_size=16384)
```
Hop budget (time between frames):   92.9 ms
Loop wall time (avg):               42.9 ms
──────────────────────────────────────────
Budget surplus:                     50.0 ms (54% free)
```

### Post-fix estimate (chunk_size=4096, GPU iFFT)
```
Hop budget (time between frames):   46.4 ms
Loop wall time (estimated):         ~6-11 ms
──────────────────────────────────────────
Budget surplus (estimated):         ~35-40 ms (75-87% free)
```

The pipeline has substantial headroom even with the smaller hop. This leaves
room to add GL rendering time (~1-5 ms) and still maintain a healthy budget.

---

## Recommended Configurations

| Config | chunk_size | overlap | Window | Hop | Max fps | Trade-off |
|---|---:|---:|---:|---:|---:|---|
| Current | 16384 | 0.75 | 371.5 ms | 92.9 ms | 10.8 | Full chromatic range, slow |
| Balanced (50ms) | 2048 | 0.50 | 46.4 ms | 23.2 ms | 43.1 | Loses ~3 lowest octaves |
| Responsive (25ms) | 2048 | 0.25 | 46.4 ms | 34.8 ms | 28.7 | Same loss, less overlap |
| DAW-like (6ms) | 512 | 0.50 | 11.6 ms | 5.8 ms | 172.3 | Only high frequencies |

Recommended starting point for the demo: **`chunk_size=4096, overlap_factor=0.5`**

```
window = 4096 / 44100 = 92.9 ms
hop    = 2048 / 44100 = 46.4 ms  →  21.5 fps
```

This brings the visualization into a range comparable to a moderately responsive
spectrogram while retaining most of the chromatic range. The CWT may need
adjustment to handle a shorter input — `max_conv_n` will also shrink.

---

## Fix Roadmap

### Fix 1 — Reduce chunk_size (APPLIED)

**Impact:** Largest improvement to perceived responsiveness
**Risk:** Low-frequency CWT accuracy degrades; lowest octaves may be invalid
**Status:** Applied in `src/subshader/config.py`

```python
chunk_size: int = 1 << 12  # 4096 (was 16384)
overlap_factor: float = 0.5  # (was 0.75)
```

Result: window=92.9 ms, hop=46.4 ms, 21.5 fps.

### Fix 2 — Move iFFT to GPU (APPLIED)

**Impact:** Reduces CWT from ~42 ms to ~5-10 ms
**Risk:** Low (algorithmic equivalence is exact; just changes device)
**Status:** Applied in `CuPyWavelet.class_specific_cwt`

Before:
```python
conv_f_gpu = input_f_gpu * self.kernel_f_bank_gpu
conv_f_cpu = cp.asnumpy(conv_f_gpu)           # download full matrix
conv_tf_cpu = ifft(conv_f_cpu, axis=1)        # CPU iFFT
return conv_tf_cpu[:, :self.input_n]
```

After:
```python
conv_f_gpu = input_f_gpu * self.kernel_f_bank_gpu
conv_tf_gpu = cp_fft.ifft(conv_f_gpu, axis=1)       # GPU iFFT
conv_tf_trimmed_gpu = conv_tf_gpu[:, :self.input_n]
return cp.asnumpy(conv_tf_trimmed_gpu)               # download only result slice
```

This eliminates the 31.5 ms CPU iFFT and replaces the 4.6 ms full-matrix download
with a proportionally smaller slice download.

### Fix 3 — Trim low-frequency octaves (future)

**Impact:** Allows smaller `chunk_size` without edge-effect corruption
**Risk:** Loses low-frequency visualization (bass notes)
**Effort:** Change `num_octaves` or `root_note_a0_hz` in `config.py`

Reducing `num_octaves` from 10 to 7 removes A0–A2 (27.5–110 Hz), but allows
`max_conv_n` to shrink dramatically, which directly enables a smaller `chunk_size`.

---

## How to Re-run the Profiler

```bash
cd /path/to/sub-shader
source venv/bin/activate
python research/pipeline_timing_profile.py --frames 64 --headless
```

With GL rendering enabled (requires display):
```bash
python research/pipeline_timing_profile.py --frames 32
```

---

## Appendix: Initialization Timing

| Stage | Time |
|---|---:|
| AudioInput init | 1.6 ms |
| CuWavelet init (GPU kernel upload) | 581.5 ms |
| Kernel bank size | 23 MB (116 × 26005 complex64) |

Startup time is dominated by GPU kernel upload (581 ms). This is a one-time
cost and does not affect frame latency. It could be reduced by caching the
kernel bank on disk if startup time becomes an issue.
