# Pre-Demo Refactor Notes

Tracking deferred cleanup from the DSP.md figure work. Surgical changes
landed during authoring; the items below get done in the v1 demo final pass
once all README/DSP.md content is locked.

---

## Context

Phase 05 added a plotting library expansion to support DSP.md figures:

- `research/utilities/plotting.py` — atomic plotters, grid scaffold, image row, single-shot CWT wrapper
- `research/utilities/dsp_helpers.py` — `BouncingChirpConfig` dataclass on top of existing builders
- `research/utilities/style.py` — atomic plotter constants section
- `research/dsp_figures.py` — DSP.md figure generators (currently §1 motivator)
- `research/test_suite.py` — `--dsp-figures` flag

The new helpers are used by `dsp_figures.py`. Existing callers
(`comparison.py`, `figures.py`) were NOT migrated to use them yet — that's
this checklist.

---

## comparison.py — delegate inline rendering to helpers

The 5×3 comparison grid in `comparison.py:282–440` duplicates logic the
helpers now own. Replace inline blocks with helper calls.

- [ ] Reference row chirp branch (`comparison.py:312–328`) → `plot_inst_freq`
- [ ] Reference row file branch (`comparison.py:330–339`) → `plot_time_series`
- [ ] DAW row (`comparison.py:347–373`) → `render_image_row`
- [ ] Spectrogram rows (`comparison.py:375–409`) → `render_spectrogram_row` (already exists, just call it)
- [ ] Inline spine setting `"#444444"` / `0.8` (multiple sites) → `style.SPINE_COLOR` / `style.SPINE_LINEWIDTH`
- [ ] Inline grid setting `"white"` / `0.08` / `0.5` (multiple sites) → `style.AXIS_GRID_COLOR` / `style.AXIS_GRID_ALPHA` / `style.AXIS_GRID_LINEWIDTH`
- [ ] Inline `99.0` percentile (L234, L253–255) → `style.VMAX_PERCENTILE`
- [ ] Inline `1.5` linewidth + `0.9` alpha (L319) → `style.INST_FREQ_LINEWIDTH` / `style.INST_FREQ_ALPHA`
- [ ] Inline `1.10` waveform y-padding (L334) → `style.WAVEFORM_YLIM_PADDING`

Expected line reduction: ~80–100 lines.

---

## figures.py — verify imports + match conventions

`figures.py` already uses the helpers correctly (`create_figure_scaffold`,
`render_top_row`, `render_spectrogram_row`). After the comparison.py
refactor:

- [ ] Verify all `from utilities import (...)` lines still resolve
- [ ] Spot-check `chirp_signal_comparison`, `polyphonic_signal_comparison`,
      `musical_signal_comparison` figures look identical to pre-refactor

---

## Optional helper extractions

Only do these if usage justifies them (more than 1–2 callers).

- [ ] Extract `render_fft_row` if a row-based caller needs FFT panels
- [ ] Extract `render_cwt_row` for non-streaming CWT row rendering

---

## Signal-gen library cleanup

- [ ] Optional: unify `research/utilities/signals.py` registry with
      synthetic generators (single registry of files + builders)
- [ ] Optional: refactor `build_X(sr, duration_s, ...)` signatures to take
      config dataclasses uniformly (currently only `BouncingChirpConfig`
      exists; would need `WanderingChirpConfig`, `FmChirpConfig`, etc.)
- [ ] Update all `build_X` callers if signatures change

---

## Verification gate before merging

- [ ] `python research/test_suite.py --timing` — completes without error
- [ ] `python research/test_suite.py --test` — all pytest tests pass
- [ ] `python research/test_suite.py --compare-methods` — figures generate
      without error; visual diff vs baseline
- [ ] `python research/test_suite.py --figures` — README figures generate;
      visual diff vs baseline
- [ ] `python research/test_suite.py --dsp-figures` — DSP.md figures
      generate; visual diff vs baseline
- [ ] Visual diff: `assets/images/generated/comparison_grid.png` before/after
- [ ] Visual diff: `assets/images/generated/chirp_signal_comparison.png` before/after

---

## Larger architectural moves (defer to post-v1)

- [ ] Consider moving `research/utilities/plotting.py` →
      `src/subshader/utilities/plotting.py` for cleaner dependency
      direction. Today `src/subshader/dsp/_figures/` would awkwardly import
      from `research/`; moving plotting.py up makes everything import from
      `subshader.utilities`.
- [ ] If multiple DSP.md figures grow beyond §1 motivator, consider
      splitting `dsp_figures.py` into `dsp_figures/__init__.py`,
      `dsp_figures/motivator.py`, `dsp_figures/<other>.py`.
