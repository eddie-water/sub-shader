---
quick_id: 260519-a1p
description: Restore figure_1 row-1 inst-freq overlay via dsplot TimeSeriesPanel twin-axis support; iterate styling; extend overlay to rows 2 and 3.
status: complete
date: 2026-05-19
branch: gsd/phase-08-codebase-refactoring-and-module-cleanup
---

# Quick Task 260519-a1p — Summary

## Goal

Restore the row-1 inst-freq overlay that the dsplot rebuild of figure_1 had dropped. Mirror the legacy `motivator.py::section1` design (chirp time-series + inst-freq curve on overlaid y-axes, Hz labels on the LEFT to visually rhyme with rows 2/3) using new dsplot framework hooks.

## Outcome

`figure_1_fourier_vs_wavelet_v8_inst_freq_on_all_rows.png` — orange inst-freq curve overlaid on **all three rows**, not just row 1. The overlay turns rows 2/3 into "STFT/CWT vs ground truth" comparisons rather than just "here's a spectrogram", which is a stronger motivator beat than the original design called for.

## Changes

### Framework (dsplot — sibling work, reusable)

- **New `Line` plottable** (`research/dsplot/plottables/line.py`) — generic 1D line plot. Color resolves lazily to `style.NEUTRAL_COLOR`; linewidth to `style.DEFAULT_DROPLINE_LINEWIDTH`; default `zorder=3` so it lands above a `TimeSeries` (zorder=2). `draw()` deliberately does NOT set xlim/ylim/facecolor — owning panel controls those, important for twin-axis use.
- **Twin-axis support on `TimeSeriesPanel`** (`research/dsplot/panels/time_series_panel.py`) — opt-in via `twin_y=True` kwarg with `twin_y_side`, `twin_y_label`, `twin_yticks`, `twin_ytick_labels`, `twin_ylim`. New `add_twin(plottable)` method. Internals mirror legacy `motivator.py` lines 355-371: `twinx()`, `tick_left()` + `set_label_position("left")` when side="left", transparent patch via `set_visible(False)`, spine styling via `style.SPINE_COLOR`.
- **New style constants** in `research/dsplot/style.py`: `INST_FREQ_COLOR`, `INST_FREQ_LINEWIDTH`, `INST_FREQ_ALPHA`.

### Figure (figure_1.py)

- Row 1: chirp time-series + inst-freq Line on LEFT-side twin axis. Drops the previous placeholder chrome (`y_label="f (Hz)"` + `yticks=[]`).
- Rows 2 and 3: inst-freq Line added directly to the heatmap panel's primary axes (shared bin-space y-axis — no twin needed).
- Bin-space conversion via `np.interp(inst_freq, cwt_freqs, np.arange(len(cwt_freqs)))`; tick labels via the same `log_f0`/`log_step` formula used by `motivator.py`.

### Visual tuning

- `INST_FREQ_COLOR`: `NEUTRAL_COLOR` → `PRIMARY_COLOR` (orange) — promotes curve from "data" to "identity"
- `INST_FREQ_LINEWIDTH`: 1.8 → 2.4 — heavier weight reads over heatmap content in rows 2/3
- Row 1 TimeSeries fill: `NEUTRAL_COLOR` → `TICK_LABEL_COLOR` (#888888) — muted gray backdrop; incidentally resolves the `fill_between` sub-pixel fade-to-empty issue on the right half of row 1 (medium gray alpha-averages into a continuous mid-gray block where bright white aliased to nothing)

## Commits (this quick task)

| Hash | Message |
|---|---|
| `bd7b079` | feat(dsplot): add Line plottable + INST_FREQ_* style constants |
| `6ca6d14` | feat(dsplot): add twin-axis support to TimeSeriesPanel |
| `49f8bf5` | feat(figure_1): restore row-1 inst-freq overlay via TimeSeriesPanel twin axis |
| `39d9e4b` | feat(figure_1): spotlight inst-freq overlay across all 3 rows |

## Deferred (captured in `[[project-figure-1]]` memory)

1. **CWT low-f "delay" diagnosis.** Below ~200 Hz the CWT magnitude peak appears AFTER the inst-freq truth line. This is intrinsic time-frequency uncertainty (Heisenberg-Gabor) — cannot be fixed by parameter tuning. Worth flagging in DSP.md §1 prose as an honest CWT limitation.
2. **CWT cone-of-influence pre-padding.** The bottom-left edge glow has two components: the intentional 22 Hz sub-trim leakage (part of locked chirp design) and an intrinsic COI artifact. The intrinsic part could be reduced by pre-padding the chirp before computing CWT, then cropping the displayed window — needs careful alignment.
3. **Possible Figure 2 — click+drone demo.** The chirp shows STFT smearing only at the 60 Hz dip (subtle). A more strikingly visual demo would be a short broadband click on top of a sustained low tone, OR a kick drum + sub-bass (most musically honest for SubShader). Together with Figure 1 it would cover both axes of the time-frequency tradeoff.

## Iteration trail (PNGs in `assets/images/generated/`)

| v | Change |
|---|---|
| v3 | Initial overlay, row 1 only, INST_FREQ_COLOR=NEUTRAL_COLOR (near-white) |
| v4 | INST_FREQ_COLOR=PRIMARY_COLOR (orange) |
| v5 | LINEWIDTH 1.8→2.4 |
| v6 | INST_FREQ_COLOR=TICK_LABEL_COLOR (gray — misread of request, reverted) |
| v7 | Inst-freq back to orange, time-series fill → TICK_LABEL_COLOR (muted gray) |
| v8 | Inst-freq extended to rows 2 and 3 — **LOCKED** |
