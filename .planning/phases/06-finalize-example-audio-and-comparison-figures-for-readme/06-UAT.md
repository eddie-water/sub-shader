---
status: diagnosed
phase: 06-finalize-example-audio-and-comparison-figures-for-readme
source: 06-01-SUMMARY.md
started: 2026-03-24T19:00:00Z
updated: 2026-03-24T19:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Bouncing Chirp Audio Playback
expected: bouncing_chirp.wav plays in DAW/Edison as a ~6s ascending chirp sweeping 20Hz-20kHz with audible parabolic dips (pitch drops then recovers) roughly 9 times across the sweep.
result: pass

### 2. Bouncing Chirp Shape in Comparison Grid
expected: In the comparison grid PNG, the bouncing chirp signal row shows a clear ascending frequency trace with visible dip-and-recover patterns — matching the hand-drawn sketch concept.
result: pass

### 3. --comparison-grid Flag Generates 3x3 Grid
expected: Running `python research/benchmark.py --comparison-grid` from project root produces comparison_grid.png with 3 signal columns x 5 rows (Reference, DAW, STFT, PyWavelet CWT, SubShader CWT).
result: pass

### 4. --dpi Flag Produces Named Variants
expected: Running `python research/benchmark.py --comparison-grid --dpi 200` produces comparison_grid_200dpi.png. The --dpi flag controls output resolution and filename suffix.
result: issue
reported: "not pass - the chirp instantaneous frequency plot doesn't match 100% with the methods below. Reference trace is misaligned with STFT/PyWavelet/SubShader spectrograms — time axis mismatch where reference continues but spectrograms have ended."
severity: minor

### 5. --timing Flag Behavior
expected: Running `python research/benchmark.py --timing` runs timing comparison across STFT, PyWavelet, and SubShader methods with timing output for each.
result: issue
reported: "not pass - it only does subshader as its default config'd - its also supposed to time the other modules like the audio player and the renderer / plotting init and runtime. --timing should stay as-is timing subshader as a whole. Need new --comparison flag that runs all method timings (STFT, PyWavelet, SubShader with avg/min/max) and produces the comparison figure. --stub-pywt should skip pywavelet and append _STUB_PYWT to filename."
severity: major

## Summary

total: 5
passed: 3
issues: 2
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "Reference instantaneous frequency plot aligns with spectrogram rows in comparison grid"
  status: failed
  reason: "User reported: chirp instantaneous frequency plot doesn't match 100% with the methods below — time axis mismatch where reference continues but spectrograms have ended"
  severity: minor
  test: 4
  root_cause: "Line 806 computes duration_s = frames_processed * chunk_size / sr which ignores overlap. With overlap_factor=0.5, this nearly doubles the true duration. Correct formula: ((frames_processed - 1) * hop_size + chunk_size) / sr. The inflated duration stretches the reference waveform/inst-freq trace past the spectrogram energy."
  artifacts:
    - path: "research/benchmark.py"
      issue: "Line 806: duration_s formula ignores frame overlap"
  missing:
    - "Fix duration_s to account for hop_size overlap"
  debug_session: ""

- truth: "benchmark.py has a --comparison flag that runs all 3 methods (STFT, PyWavelet, SubShader) with timing stats (avg/min/max) and produces the comparison figure"
  status: failed
  reason: "User reported: --timing only does subshader pipeline. Need --comparison flag that runs all method timings and produces comparison plot. --timing stays as-is for subshader-only pipeline timing. --stub-pywt skips pywavelet and appends _STUB_PYWT to filename."
  severity: major
  test: 5
  root_cause: "generate_comparison_grid() already runs all 3 methods via time_call() but discards timing values. --comparison flag needs to: (1) collect timing arrays per method, (2) print stats via existing print_results_header/print_results_row helpers, (3) produce the grid figure. ~40-60 lines of changes to benchmark.py."
  artifacts:
    - path: "research/benchmark.py"
      issue: "generate_comparison_grid discards timing; no --comparison flag exists"
  missing:
    - "Add --comparison argparse flag"
    - "Wrap STFT call in time_call(), collect timing arrays instead of discarding"
    - "Print per-signal timing stats after processing"
    - "Use _STUB_PYWT suffix (not _STUB) when --stub-pywt is active"
  debug_session: ""
