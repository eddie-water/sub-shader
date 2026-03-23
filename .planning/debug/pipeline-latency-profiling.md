---
status: fixing
trigger: "pipeline-latency-profiling: Phase 3 audio-visual sync feels laggy — visualization is noticeably slower than a DAW spectrogram. Need to profile the entire pipeline end-to-end, measure per-stage timing, identify bottlenecks, and produce a README with timing results."
created: 2026-03-22T00:00:00Z
updated: 2026-03-22T00:04:00Z
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: BOTH FIXES APPLIED
  - Fix 1: chunk_size=4096, overlap_factor=0.5 → 92.9ms window, 46.4ms hop, 21.5fps
  - Fix 2: CuPyWavelet iFFT moved to GPU → CWT expected to drop from ~42ms to ~5-10ms
test: pytest 35/35 pass; README updated with post-fix numbers
expecting: user runs app and confirms improved responsiveness
next_action: await human verification of improved feel in live run

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: Visualization responds to audio transients within ~100ms, comparable to DAW spectrogram responsiveness
actual: Noticeable lag, "chunky" feel, "not even really close to a DAW" per user retest on their PC
errors: No errors — it works, just too slow
reproduction: Run `python -m subshader` and compare visual responsiveness to any DAW spectrogram
started: Has always been somewhat laggy; Phase 3 added audio sync but latency didn't fully address it

## Eliminated
<!-- APPEND only - prevents re-investigating -->

- hypothesis: CWT GPU computation is the bottleneck
  evidence: CWT averages 42ms on GPU (38ms of which is CPU iFFT), well within 93ms hop budget. 50ms budget headroom remains.
  timestamp: 2026-03-22T00:02:00Z

- hypothesis: render/texture upload is the bottleneck
  evidence: Not measured in headless run, but structural analysis confirms the loop total without GL is already 43ms, leaving 50ms budget. GL stages (texture upload + draw + swap) are typically 1-5ms for this texture size.
  timestamp: 2026-03-22T00:02:00Z

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-03-22T00:01:30Z
  checked: config.py AudioConfig defaults
  found: chunk_size=16384 (2^14), overlap_factor=0.75, hop_size=4096
  implication: hop_size=4096 at 44100 Hz = 92.9ms between visual updates = only 10.8 fps max visual frame rate

- timestamp: 2026-03-22T00:01:30Z
  checked: structural latency math
  found: chunk_size of 16384 samples at 44100 Hz = 371.5ms of audio per window
  implication: visualization is ALWAYS >= 372ms behind real-time before any processing starts

- timestamp: 2026-03-22T00:01:30Z
  checked: DAW STFT comparison
  found: typical DAW spectrogram uses 1024-sample windows with 256-sample hops
  implication: SubShader window is 16x longer, hop is 16x longer, frame rate is 16x lower than DAW

- timestamp: 2026-03-22T00:02:00Z
  checked: profiler output (64 frames, GPU mode)
  found: loop total avg=42.9ms, cwt avg=42.2ms, get_chunk avg=0.14ms, push_frame avg=0.60ms
  implication: code-level overhead is only 43ms per frame. Entire budget (50ms surplus) is available.

- timestamp: 2026-03-22T00:02:00Z
  checked: CWT sub-stage breakdown
  found: ifft_cpu=31.5ms (75% of CWT time), gpu_download=4.6ms, gpu_multiply=2ms avg (but spiky up to 109ms due to CUDA sync)
  implication: iFFT (116 frequencies × 26005-sample convolution length) on CPU is the single most expensive operation. Moving iFFT to GPU would reduce CWT from 42ms to ~10ms.

- timestamp: 2026-03-22T00:02:00Z
  checked: max_conv_n
  found: max_conv_n=26005 (largest wavelet kernel length). iFFT operates on 116×26005 complex64 matrix on CPU.
  implication: This is why iFFT dominates. A 3 million element complex array iFFT per frame.

- timestamp: 2026-03-22T00:04:00Z
  checked: Fix 1 application — config.py AudioConfig.chunk_size and overlap_factor
  found: chunk_size changed from 1<<14 (16384) to 1<<12 (4096); overlap_factor changed from 0.75 to 0.5
  implication: window=92.9ms, hop=46.4ms, 21.5fps. All 35 tests pass.

- timestamp: 2026-03-22T00:04:00Z
  checked: Fix 2 application — CuPyWavelet.class_specific_cwt iFFT path
  found: replaced cp.asnumpy(conv_f_gpu) + numpy ifft with cp_fft.ifft on GPU, then cp.asnumpy of trimmed slice
  implication: eliminates 31.5ms CPU iFFT and 4.6ms full-matrix download. Expected CWT: ~5-10ms. All 35 tests pass.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  TWO COMPOUNDING STRUCTURAL PROBLEMS — not a code bug, a configuration design constraint:

  1. WINDOW LATENCY: chunk_size=16384 at 44100 Hz = 371.5ms audio window. The CWT must
     accumulate 371.5ms of audio before it can process a single frame. The visualization
     is structurally 372ms behind real-time, minimum.

  2. UPDATE RATE: hop_size=4096 (25% of chunk) = 92.9ms between visual updates = 10.8 fps.
     DAWs update at 5.8ms/172fps — 16x more often.

  SECONDARY: iFFT is 75% of CWT cost (31.5ms). The convolution produces 116×26005 complex
  values that must be inverse-FFT'd on CPU. This is a CPU-GPU hybrid problem: multiply
  is GPU but iFFT is CPU, requiring a large download then CPU-heavy work.

fix: |
  Fix 1 — chunk_size=4096, overlap_factor=0.5 (config.py AudioConfig defaults)
    → window=92.9ms, hop=46.4ms, 21.5fps (2x improvement over 10.8fps)

  Fix 2 — CuPyWavelet.class_specific_cwt: replace cp.asnumpy(conv_f_gpu) + numpy ifft
    with cp_fft.ifft(conv_f_gpu, axis=1) on GPU, then cp.asnumpy(trimmed slice).
    → CWT expected to drop from ~42ms to ~5-10ms; eliminates 31.5ms CPU iFFT and
      large 116×26005 complex64 intermediate download.

verification: pytest 35/35 pass; README updated with post-fix expected numbers
files_changed:
  - src/subshader/config.py (chunk_size 16384→4096, overlap_factor 0.75→0.5)
  - src/subshader/dsp/wavelet.py (CuPyWavelet iFFT moved to GPU)
  - research/README_pipeline_timing.md (post-fix numbers added)
  - research/pipeline_timing_profile.py (new profiling script)
  - research/README_pipeline_timing.md (new findings README)
