# Test Suite Design

## test_suite.py — CLI Dispatcher

```
python research/test_suite.py --timing
python research/test_suite.py --test
python research/test_suite.py --compare-methods
python research/test_suite.py --figures
```

---

## Modes

### 1. `--timing` — Profile the pipeline

Run SubShader end-to-end on one audio file. Time every stage.

```python
config = PipelineConfig(file_path="assets/audio/reference/beltran_sc_rip.wav")
audio = AudioStream(config)
dsp = GpuCWT(config)          # or CpuCWT fallback
renderer = Renderer(config, dsp.output_shape)

for i in range(num_frames):
    chunk = audio.get_chunk()
    coefs = dsp.process(chunk)       # @timed attributes available
    renderer.update(coefs)           # timed

# Print results to terminal
# Write results to assets/timing/YYYYMMDD_HHMMSS_timing.txt
```

**Input:** One audio file (default: beltran full rip, overridable via flag)
**Output:** Terminal table + timestamped results file

### 2. `--test` — Run pytest

```
pytest research/tests/ -v
```

Some tests produce figures (stored in assets). Most just assert correctness.

**Input:** Test fixtures
**Output:** Terminal results, optionally generated figures

### 3. `--compare-methods` — Compare DSP backends on one signal

Produces one figure per signal. Each figure has rows:

```
┌─────────────────────────────────┐
│  Row label  │   Signal Name     │
├─────────────┼───────────────────┤
│ Time Series │ [waveform plot]   │
│ DAW Ref     │ [edison image]    │
│             │ or "No Reference  │
│             │  Image Found —    │
│             │  Place in assets/ │
│             │  images/reference/│
│             │  and rerun"       │
│ STFT        │ [spectrogram]     │
│ PyWavelet   │ [spectrogram]     │
│ SubShader   │ [spectrogram]     │
└─────────────┴───────────────────┘
```

```python
config = PipelineConfig(file_path="assets/audio/reference/bouncing_chirp.wav")
audio = AudioStream(config)

stft = STFT(config)
pywt = PywaveletCWT(config)
cwt  = CpuCWT(config)

# Process all chunks through each backend
# Render figure with row labels + padding
# Save to assets/images/generated/chirp_comparison.png
```

Run once per signal. Default runs all three:

```
python research/test_suite.py --compare-methods                    # all 3 signals
python research/test_suite.py --compare-methods --signal chirp     # just chirp
```

**Input:** Audio file + DAW reference image (optional, graceful stub if missing)
**Output:** One PNG per signal → `assets/images/generated/`

### 4. `--figures` — Generate all documentation images

Runs `--compare-methods` for all signals, plus any other doc figures (timing bar chart, etc.).

```python
# Generate per-signal comparison figures (reuses --compare-methods code)
for signal in [chirp, polyphonic, musical]:
    generate_method_comparison(signal)

# Generate timing bar chart
generate_timing_bar_chart()

# Optional: generate 5x3 composite grid (utility kept, not default)
# generate_comparison_grid()   # uncomment if needed
```

**Output:** All PNGs in `assets/images/generated/`

README uses the three per-signal figures side by side:

```html
<p align="center">
  <img src="assets/images/generated/chirp_comparison.png" width="32%">
  <img src="assets/images/generated/polyphonic_comparison.png" width="32%">
  <img src="assets/images/generated/musical_comparison.png" width="32%">
</p>
```

The 5×3 grid utility function stays in `comparison.py` for when you want the composite. Not deleted, just not the default README figure anymore.

---

## Signal Registry

Instead of hardcoded signal definitions scattered across files:

```python
# research/utilities/signals.py

SIGNALS = [
    {
        "name": "chirp",
        "label": "Bouncing Chirp",
        "audio": "assets/audio/generated/bouncing_chirp.wav",
        "reference": "assets/images/reference/bouncing_chirp_edison.png",
        "type": "synthetic",   # generated at runtime
    },
    {
        "name": "polyphonic",
        "label": "MIDI Sine Waves",
        "audio": "assets/audio/reference/midi_sine_waves.wav",
        "reference": "assets/images/reference/midi_sine_wave_edison.png",
        "type": "file",
    },
    {
        "name": "musical",
        "label": "Beltran (4 Bars)",
        "audio": "assets/audio/reference/beltran_sc_rip_4_bar.wav",
        "reference": "assets/images/reference/beltran_4_bar_edison.png",
        "type": "file",
    },
]
```

`--compare-methods` iterates this list. Adding a new signal = append to the list + drop the audio file in reference/.

---

## Utilities

Shared helpers that all modes use:

```
research/utilities/
├── style.py          visual constants (colors, fonts, sizes, spacing)
├── plotting.py       figure scaffold, row renderers, heatmap helpers
├── signals.py        SIGNALS registry + synthetic signal builders
├── printing.py       terminal table formatting
├── timing.py         time_call, TimingAccumulator
├── constants.py      paths, output dirs, DSP params
└── wav_export.py     signal → WAV file
```

These enforce consistent style across all figures. Any test or mode that generates a figure uses `style.py` constants and `plotting.py` renderers. Override by passing params — defaults come from style.

---

## Assets (follows test suite structure)

```
assets/
├── audio/
│   ├── reference/                     committed, inputs
│   │   ├── beltran_sc_rip.wav         runtime default + timing
│   │   ├── beltran_sc_rip_4_bar.wav   --compare-methods (musical)
│   │   ├── beltran_sc_rip_8_bar.wav   --timing (alt)
│   │   ├── beltran_sc_rip_16_bar.wav  --timing (alt)
│   │   ├── midi_sine_waves.wav        --compare-methods (polyphonic)
│   │   └── a2a3_a4_minor_scale.wav    --test fixture
│   └── generated/                     created by test_suite.py
│       └── bouncing_chirp.wav         --compare-methods (chirp, synthesized)
│
├── images/
│   ├── reference/                     committed, inputs
│   │   ├── bouncing_chirp_edison.png  --compare-methods DAW row
│   │   ├── midi_sine_wave_edison.png  --compare-methods DAW row
│   │   ├── beltran_4_bar_edison.png   --compare-methods DAW row
│   │   └── numpy_vs_cupy_diff.png     DSP.md
│   └── generated/                     created by test_suite.py
│       ├── chirp_comparison.png       --compare-methods / --figures
│       ├── polyphonic_comparison.png  --compare-methods / --figures
│       ├── musical_comparison.png     --compare-methods / --figures
│       ├── comparison_grid.png        --figures (composite, optional)
│       ├── timing_bar_chart.png       --figures
│       ├── dpi/                       DPI variants
│       └── stubs/                     fast iteration
│
├── timing/                            created by --timing
│   └── 20260406_143022_timing.txt     timestamped results
│
├── plots/                             architecture diagrams
│
└── archive/                           old/unused files
```
