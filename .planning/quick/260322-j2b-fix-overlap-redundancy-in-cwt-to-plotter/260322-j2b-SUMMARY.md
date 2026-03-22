---
phase: quick
plan: 260322-j2b
subsystem: dsp
tags: [cwt, overlap, hop-center, wavelet, diagnostic]
dependency_graph:
  requires: []
  provides: [Wavelet.extract_hop_center, overlap_factor wiring]
  affects: [src/subshader/dsp/wavelet.py, src/subshader/__main__.py, research/overlap_diagnostic.py]
tech_stack:
  added: []
  patterns: [hop-center extraction, overlap_factor propagation through constructor chain]
key_files:
  created:
    - research/overlap_diagnostic.py
    - assets/images/diagnostics/overlap_redundancy_diagnostic.png
  modified:
    - src/subshader/dsp/wavelet.py
    - src/subshader/__main__.py
decisions:
  - "extract_hop_center placed between discard_unreliable_coefs and downsample in cwt() pipeline — reliable region is the natural input, and downsampling from a smaller hop-center window still produces target_width columns"
  - "overlap_factor default 0.0 ensures backward compatibility — existing callers without the parameter see no change in behavior"
  - "hop_fraction = 1 - overlap_factor applied to reliable_width, not input_n — the reliable region is what the downstream pipeline sees; its center fraction matches the audio hop fraction"
metrics:
  duration: ~15 minutes
  completed: 2026-03-22T17:51:14Z
  tasks_completed: 2
  tasks_total: 2
  files_changed: 4
---

# Quick Task 260322-j2b: Fix Overlap Redundancy in CWT-to-Plotter Summary

**One-liner:** Hop-center extraction added to Wavelet.cwt() pipeline — consecutive frames now tile without redundant overlapping wings at overlap_factor=0.75.

## What Was Built

### Diagnostic Script (research/overlap_diagnostic.py)
Standalone script that proves the overlap redundancy problem and validates the fix. Generates a synthetic 440-880 Hz chirp, extracts 5 consecutive overlapping chunks, and runs NumPyWavelet CWT on each. Produces a matplotlib figure (`assets/images/diagnostics/overlap_redundancy_diagnostic.png`) with two columns per overlap factor:
- Left: full reliable region downsampled to target_width (current/old behavior — shows repeated wing content across frame boundaries)
- Right: hop-center only downsampled to target_width (fix — each frame contains only its unique time content)

Summary table output:
```
overlap_factor   hop_size  reliable_width  center_width  frame_count
----------------------------------------------------------------------
          0.00      16384            8192          8192            5
          0.50       8192            8192          4096            5
          0.75       4096            8192          2048            5
```

### Production Fix (src/subshader/dsp/wavelet.py)
- `Wavelet.__init__` gains `overlap_factor: float = 0.0` parameter — backward compatible
- `Wavelet.extract_hop_center(reliable_coefs)` concrete method extracts the center `(1 - overlap_factor)` fraction of reliable coefficients:
  - Returns input unchanged when `overlap_factor <= 0`
  - Uses `max(1, int(reliable_width * hop_fraction))` to guard minimum width
  - Center is computed symmetrically: `center_start = (reliable_width - center_width) // 2`
- `Wavelet.cwt()` pipeline updated: `discard_unreliable_coefs` -> `extract_hop_center` -> `downsample`
- `overlap_factor` propagated through `AntsWavelet`, `NumPyWavelet`, `CuPyWavelet`, `PyWavelet` constructors
- Alias classes `NpWavelet` and `CuWavelet` inherit without `__init__` override, so they pass through automatically

### Wiring (src/subshader/__main__.py)
`config.audio.overlap_factor` (default 0.75) now passed to wavelet constructor, so the production pipeline benefits from the fix immediately.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `python research/overlap_diagnostic.py` ran without error, produced PNG
- Verification check confirmed `result.shape == (116, 256)` and shape identity between `overlap_factor=0.0` and `overlap_factor=0.75`
- Task 3 (visual checkpoint): auto-approved (auto_advance=true)

## Known Stubs

None.

## Self-Check: PASSED

- research/overlap_diagnostic.py: FOUND
- assets/images/diagnostics/overlap_redundancy_diagnostic.png: FOUND
- SUMMARY.md: FOUND
- Commit be22b76 (Task 1): FOUND
- Commit 5917177 (Task 2): FOUND
