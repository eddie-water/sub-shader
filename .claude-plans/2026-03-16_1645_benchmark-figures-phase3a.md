# SubShader Benchmark Figures — Phase 3a Visual Polish

**Date**: 2026-03-16 16:45
**Status**: In Progress
**Context**: Finalizing visual style for 3 comparison figures (Chirp, Polyphonic, Musical)

---

## Completed Work (Phases 0-2)

### Phase 0: Progress Bar Cleanup ✓
- Single `\r` progress line (overwrites in place)
- Split output into `print_figure_header()` (before) + `print_figure_results()` (after)
- Clean separator-based formatting with timing table

### Phase 1: Figure Layout ✓
- **Chirp**: 4 rows — Instantaneous Frequency + 3 spectrograms (waveform removed)
- **Polyphonic**: 5 rows — MIDI Piano Roll + Edison DAW + 3 spectrograms
- **Musical**: 5 rows — Audio Waveform + Edison DAW + 3 spectrograms
- Auto-detect frame count from audio file (158 frames polyphonic, 159 frames musical)
- Switched to `assets/audio/daw/musical_audio_example.wav` (30s, was beltran 208s)

### Phase 2: Visual Style ✓
- **Typography**: Suptitle 42pt, Subplot titles 24pt, Y-tick labels 16pt
- **Axes**: Y-ticks at [100, 1k, 10k, 20k] Hz log scale, no ylabel
- **Layout**: Equal 6% margins left/right, removed `bbox_inches="tight"`
- **Removed**: Suptitle timing subtitle (moved to per-subplot titles)

### Stub System
```bash
py research/benchmark.py --stub  # Instant render with random noise (no DSP)
```
- Saves to `assets/images/benchmarks/stubs/` with `_STUB` suffix
- Uses real reference images (MIDI, DAW screenshots, waveforms)
- Allows instant visual iteration without 10+ min waits

---

## Phase 3a: Visual Impression & Polish

### Discussion Questions (Answer these before moving to Phase 3b)

**1. Overall Balance**
- Do the 3 methods (STFT, PyWavelet, SubShader) feel equally weighted?
- Or does one dominate visually?
- Should they be scaled/sized differently to emphasize performance differences?

**2. Reference vs. Computed**
- Top rows (MIDI, DAW, waveform) have different visual style than magma spectrograms
- Is this separation intentional/good?
- Should reference rows integrate better aesthetically?
- Should reference rows be muted/greyscale to recede?

**3. Reading Flow**
- Does the eye naturally parse: "here's the reference → here are 3 methods being compared"?
- Is the hierarchy clear?
- Does viewer know which is "best" or are they equally presented?

**4. Color & Contrast**
- Magma on black background feels dark/muted
- Alternatives: inferno (yellows/oranges), viridis (blue/green), plasma (purple/pink)
- Which feels better for spectrograms?
- Should reference images (DAW, MIDI) have any color treatment?

**5. Title Hierarchy**
- Big suptitle (42pt) + big subplot titles (24pt) feels strong
- Is it too much visual weight in titles?
- Should subtitle be re-added below suptitle (timing info)?

### Phase 3a Decisions (Fill in after discussion)

```
Overall balance:     [ ]
Reference style:     [ ]
Visual integration:  [ ]
Colormap choice:     [ ]
Title hierarchy:     [ ]
```

---

## Remaining Phases (After Phase 3a)

### Phase 3b: Detailed Polish
- [ ] Timing table placement (per-subplot or separate panel?)
- [ ] Figure dimensions & DPI optimization
- [ ] Dual-scale figures (linear + log frequency variants)
- [ ] Font choice (currently matplotlib default, consider 'DejaVu Sans')
- [ ] Axis tick label formatting (currently "100", "1k", "10k", "20k")

### Phase 4: Seaborn Port
- Extract figure data layer (spectrograms, timing, metadata)
- Create parallel `_render_seaborn()` using `sns.heatmap` instead of `imshow`
- Output: `assets/images/benchmarks/seaborn/` variants
- Architecture: `_generate_comparison_figure()` → returns FigureData → `_render_matplotlib()` or `_render_seaborn()`

### Phase 5: README Integration
- Embed figures into top-level README.md
- Add performance summary text (timing table formatted for markdown)
- Link to full benchmark results
- Update description of SubShader advantages

---

## Files & Commands

### Key Files
- `research/benchmark.py` — main figure/benchmark generation
- `research/benchmark_utilities.py` — output formatting (headers, separators, timing tables)
- `research/benchmark.py` — constants: audio paths, image paths, colormap

### Quick Commands
```bash
# Instant preview (random noise, real references, no DSP wait)
py research/benchmark.py --stub

# Full render (10-15 min per figure)
py research/benchmark.py --figures

# With seaborn variants
py research/benchmark.py --figures --seaborn

# Timing instrumentation
py research/benchmark.py --timing

# All modes
py research/benchmark.py --all
```

### Current Figure Locations
```
assets/images/benchmarks/
  ├── chirp_signal_comparison.png           # Real render
  ├── polyphonic_signal_comparison.png      # Real render
  ├── musical_signal_comparison.png         # Real render
  └── stubs/
      ├── chirp_signal_comparison_STUB.png
      ├── polyphonic_signal_comparison_STUB.png
      └── musical_signal_comparison_STUB.png
```

---

## Architecture Summary

### Figure Generation Pipeline
```
for each figure (chirp, polyphonic, musical):
  print_figure_header(title)              # Title block before processing

  for each frame:
    compute STFT, PyWavelet, SubShader
    live_progress(frame, total)           # Single \r line updates

  print_figure_results(timing)            # Results table after processing
  _generate_comparison_figure(...)        # Render matplotlib figure
    ├── create gridspec with variable rows (n_top + 3 spectrogram rows)
    ├── render top rows (waveform/freq_line/image)
    ├── render 3 spectrogram rows (STFT, PyWavelet, SubShader)
    └── save to PNG
```

### Layout Configuration
Each figure defined by `top_rows` descriptors:
```python
top_rows = [
  {"type": "waveform", "title": "Audio Waveform"},
  {"type": "image", "path": "...", "title": "DAW Spectrogram"},
]
# Renders as 5-row figure (2 top + 3 spectrogram)
```

---

## Notes for Next Session

- Stub system is working well for rapid iteration — use it to answer Phase 3a questions
- After Phase 3a decisions, move to Phase 3b (detailed polish, timing table placement)
- Phase 4 (seaborn) can happen in parallel or after Phase 3
- Phase 5 (README) depends on final figures being approved
