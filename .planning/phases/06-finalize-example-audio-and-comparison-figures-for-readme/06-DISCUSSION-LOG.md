# Phase 6: Finalize Example Audio and Comparison Figures for README - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-24
**Phase:** 06-finalize-example-audio-and-comparison-figures-for-readme
**Areas discussed:** Audio signal selection, Figure polish level, README visual flow, Scope boundary

---

## Audio Signal Selection

### Chirp column audio file

| Option | Description | Selected |
|--------|-------------|----------|
| chirp_comparison_grid.wav | Already generated for this purpose in Phase 5. 10 second linear sweep. | |
| chirp_random_walk.wav | Random frequency walk — more dynamic, shows non-stationary tracking better | |
| New chirp file | Generate something different | ✓ |

**User's choice:** New chirp file — bouncing chirp similar to hand-drawn sketch (assets/images/claude/bouncing_chirp.png) but mathematically clean.

### Bouncing chirp parameters

| Option | Description | Selected |
|--------|-------------|----------|
| 4-5 bounces, 200Hz-8kHz | Matches drawing (~4 arcs). CWT advantages most visible. | |
| 3 bounces, 100Hz-10kHz | Fewer bounces, wider range. | |
| 6+ bounces, 200Hz-12kHz | More bounces, denser motion. | |
| You decide | Claude picks best parameters | |

**User's choice:** Custom — logarithmic bounce pattern starting low: 20 → 200 → 2000 → 20k. Will confirm by running stub figure generator to iterate visually.

### Polyphonic column audio file

| Option | Description | Selected |
|--------|-------------|----------|
| overlapping_A3_A4_A5.wav | MIDI-generated overlapping notes. Clean, controlled. | |
| polyphonic_audio_example.wav | Existing polyphonic example. | ✓ |
| New polyphonic file | Generate or record something different | |

**User's choice:** polyphonic_audio_example.wav

### Musical column audio file

| Option | Description | Selected |
|--------|-------------|----------|
| beltran_sc_rip_8_bar.wav | 8-bar excerpt with percussion + bass. | ✓ |
| musical_audio_example.wav | Existing musical example. | |
| beltran_sc_rip_16_bar.wav | 16-bar version. Longer. | |
| New musical file | Different track or excerpt | |

**User's choice:** beltran_sc_rip_8_bar.wav

---

## Figure Polish Level

### Colormap

| Option | Description | Selected |
|--------|-------------|----------|
| Current colormap | Keep whatever benchmark.py currently uses | ✓ |
| Match the shader | Use same colormap as live visualization | |
| You decide | Claude picks for readability | |

**User's choice:** Current colormap (no change)

### Axis labels and titles

| Option | Description | Selected |
|--------|-------------|----------|
| Row/column headers only | Labels on edges, no per-subplot axes | |
| Minimal per-subplot axes | Edge subplots get frequency/time ticks | |
| Full labels on every subplot | Each subplot gets own labels | |
| You decide | Claude picks cleanest | |

**User's choice:** Keep as-is, defer until audio examples finalized. Mentioned wanting decorator utility functions (vertical lines, grid) for accuracy verification.

### DPI and output size

| Option | Description | Selected |
|--------|-------------|----------|
| 150 DPI, wide format | Good for README, ~1800px | |
| 300 DPI, publication quality | Sharp at any zoom, larger file | |
| You decide | Claude picks based on tradeoffs | |

**User's choice:** Generate at 4 DPIs (150, 200, 250, 300) and compare visually. Pick based on quality vs file size.

---

## README Visual Flow

### Grid vs individual figures

| Option | Description | Selected |
|--------|-------------|----------|
| Grid only | One big comparison grid. Per-signal sections get captions only. | ✓ |
| Grid + individual figures | Grid as hero + standalone per-signal figures | |
| You decide | Claude picks based on length/density | |

**User's choice:** Top-level README is a landing page — one comparison grid. Deep analysis in DSP.md submodule README.
**Notes:** "There should be one, or one small set of comparisons that highlight the performance and advantages of subshader against common DSP methods"

### Figure sizing in README

| Option | Description | Selected |
|--------|-------------|----------|
| Full width | Maximum impact | |
| Centered at ~80% | Slightly inset, intentional feel | ✓ |
| Centered at ~50-60% | Compact, may lose detail | |

**User's choice:** Centered at ~80%. Figure generated wide so it reads well in README and is detailed when opened in new tab.

### REWRITE markers for per-signal captions

| Option | Description | Selected |
|--------|-------------|----------|
| Leave REWRITE markers | Phase 5 flagged for user authoring | |
| Replace with scaffold captions | Claude writes placeholders for user to rewrite | ✓ |
| Remove per-signal sections | Grid tells the whole story | |

**User's choice:** Replace with scaffold captions

### Timing bar chart placement

| Option | Description | Selected |
|--------|-------------|----------|
| Keep in README | Speed is part of top-level pitch | |
| Move to DSP.md | Top-level stays visual-only | ✓ |
| Keep but regenerate | Keep in README with updated styling | |

**User's choice:** Move to DSP.md AND regenerate with updated data/styling to match comparison grid polish level.

---

## Scope Boundary

### Phase 5 plan 05-04 (foundations figures)

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 6 includes it | Fold all figure work into one phase | |
| Finish 05-04 separately first | Execute as Phase 5 before starting Phase 6 | ✓ |
| Skip 05-04 entirely | Not needed for Demo Ready | |

**User's choice:** Finish after — go back to Phase 5 after Phase 6.

### Bouncing chirp audio generation

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 6 generates it | benchmark.py flag to synthesize | |
| I'll create it myself | User provides WAV | |
| Use fm_sine.py | Extend existing script | |

**User's choice:** Ideate with Claude in Phase 6. Clean up the flag in a later phase.

### Decorator utility functions

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, include decorators | Build grid line/marker utilities | |
| Just the figures | Focus on audio + grid + README only | ✓ |

**User's choice:** Just the figures. Decorators are out of scope.

---

## Claude's Discretion

- Bouncing chirp synthesis implementation details
- Scaffold caption wording
- Timing bar chart restructuring for DSP.md
- Subplot spacing and layout

## Deferred Ideas

- Decorator utility functions (grid lines, vertical markers) — future phase
- Axis label/title styling — deferred until audio confirmed
- 05-04 foundations figures — Phase 5 work after Phase 6
- Permanent benchmark.py flag for bouncing chirp — later phase
