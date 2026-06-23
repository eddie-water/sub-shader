---
phase: quick-260521-tpz
plan: 01
subsystem: research/dsplot/figures
tags: [dsplot, sample_template, figures, suptitle_panel, showcase, notebook, three-mode-contract]
status: AWAITING_HUMAN_VERIFY
key-files:
  created:
    - research/dsplot/figures/sample_template.py
    - assets/images/dsp/figures/sample_template/v1.png
    - .planning/quick/260521-tpz-master-sample-template-figure-exercising/260521-tpz-SUMMARY.md
  modified:
    - research/dsplot/figures/__init__.py
    - src/subshader/dsp/dsp.ipynb
decisions:
  - Spotlight mode="band" not supported (valid modes are rectangle/scatter/glow); row 3 col 3 uses mode="scatter" anchored at the sine peak per the plan's fallback rule
  - Used scipy-free iterated-log chirp builder inlined into sample_template.py rather than importing utilities.build_waypoint_chirp — keeps the showcase self-contained and avoids a second `from utilities import` worldview hop for a placeholder signal
  - Re-exported sample_template (not just `from .sample_template import show as sample_template`) so callers can do `from dsplot.figures import sample_template` and then `sample_template.render()`, `sample_template.show()`, or `sample_template.embed()` — matches the three-mode contract docstring
metrics:
  duration: "~12 min"
  tasks_executed: 3 of 3 auto + checkpoint pending
  files_created: 3
  files_modified: 2
---

# Quick Task 260521-tpz: Master Sample Template Figure Summary

Final deliverable of the 3-quick dsplot batch (Q1 cleanup + Q2 SuptitlePanel/overlays → Q3 master template). New `research/dsplot/figures/sample_template.py` is a kitchen-sink dsplot showcase exercising every Panel kind, every Plottable, the new explicit-SuptitlePanel composition path, the Q2 cross-type overlay contracts, and variable cell spans — composed via `Figure.compose` with `show_cell_borders=True`. The `render() / show() / embed(target)` three-mode invocation contract is implemented in sample_template AND documented as the canonical convention for every future figure module in `research/dsplot/figures/__init__.py`.

## Absolute PNG Path

`/home/eddie-water/dev/python/sub-shader/assets/images/dsp/figures/sample_template/v1.png` (572,648 bytes)

To iterate the visual style, bump the filename: `sample_template.render(output_filename="v2.png")` → `v3.png` → `vN.png`.

## Smoke-test Output

```
$ python -c "import sys; sys.path.insert(0, 'research'); from dsplot.figures import sample_template; sample_template.render(); print('A ok')"
A ok

$ python -c "from research.dsplot.figures.sample_template import render; print(render()); print('B ok')"
/home/eddie-water/dev/python/sub-shader/assets/images/dsp/figures/sample_template/v1.png
B ok
```

Both worldviews import and render cleanly. The `[[project-dsplot-import-worldviews]]` rule is honored: `sample_template.py` uses `from .. import …` for dsplot-internal references (no `from research.X import` anywhere).

## Coverage Audit

Every Panel kind and every Plottable is exercised at least once:

| Panel kind        | Where in showcase                                                 |
|-------------------|-------------------------------------------------------------------|
| StaticPanel       | row 1 col 1 (2D Vector); row 3 col 3 (TimeSeries + Spotlight)     |
| StaticPanel3D     | row 2 col 1 (3D Vector + dashed component staircase)              |
| TimeSeriesPanel   | row 1 col 2 (twin-axis Line overlay); inside composite row 2 col 2 |
| HeatmapPanel      | row 1 col 3 (Line overlay in bin-space); row 2 col 3 (gaussian)   |
| CompositePanel    | row 2 col 2 (Stem + Dropline + Spotlight + Annotation)            |
| DynamicPanel      | row 3 col 1 (single-frame static export)                          |
| InteractivePanel  | row 3 col 2 (single-frame static export, slider bar visible)      |
| TextPanel         | via SuptitlePanel subclass in row 0                               |
| SuptitlePanel     | row 0 (explicit `rows=[[SuptitlePanel(...)], …]` composition path) |

| Plottable          | Where in showcase                                                |
|--------------------|------------------------------------------------------------------|
| TimeSeries         | row 1 col 2; row 3 col 1 (dynamic frame); row 3 col 3            |
| Heatmap            | row 1 col 3; row 2 col 3; row 3 col 2 (interactive frame)        |
| Line               | row 1 col 2 (twin-axis); row 1 col 3 (heatmap overlay, bin-space) |
| Vector             | row 1 col 1 (2D); row 2 col 1 (3D)                                |
| VectorComponents   | row 1 col 1                                                       |
| Annotation         | inside composite row 2 col 2 ("peak" callout with arrow)         |
| Dropline           | inside composite row 2 col 2 (peak → 0 baseline)                 |
| Spotlight          | inside composite row 2 col 2 (mode="scatter"); row 3 col 3 (mode="scatter") |
| Stem               | inside composite row 2 col 2                                      |

Layout: four rows × 6-unit width (row 0 = SuptitlePanel spanning the full 6 units; rows 1–3 each = 1 + 2 + 3 or 1 + 4 + 1 or 2 + 2 + 2 = 6).

## ipynb Integration

Cell appended successfully to `src/subshader/dsp/dsp.ipynb` (cell 17 of 17). The risk-gate checks all passed: `json.loads` succeeded, no `metadata.widgets` block exists at the top level, no `.dsp.ipynb~` lock file present.

New cell:
```python
# dsplot — Sample Template (kitchen-sink showcase)
from dsplot.figures import sample_template
sample_template.show()
```

If the cell needs to be removed or moved, it's tagged with `id: dsplot-sample-template-showcase` for easy lookup.

**Fallback snippet** (in case the user wants to re-add manually after a different ipynb workflow):

> Paste into a new cell in `src/subshader/dsp/dsp.ipynb`:
>
> ```python
> from dsplot.figures import sample_template
> sample_template.show()
> ```

## Files NOT Committed

Per `[[feedback-no-auto-commits]]`, all changes are left as working-tree edits for user review. `HEAD` is still at `a618f8a` (the SuptitlePanel + Heatmap alpha commit from Q2).

**Modified:**
- `research/dsplot/figures/__init__.py` — added three-mode contract paragraph + `sample_template` re-export + `__all__` entry
- `src/subshader/dsp/dsp.ipynb` — appended `sample_template.show()` cell (cell 17, id `dsplot-sample-template-showcase`)

**New:**
- `research/dsplot/figures/sample_template.py` — kitchen-sink showcase module
- `assets/images/dsp/figures/sample_template/v1.png` — rendered PNG (572 KB)
- `.planning/quick/260521-tpz-master-sample-template-figure-exercising/260521-tpz-SUMMARY.md` — this file

**Suggested single commit message** (for the user to use at their discretion):

```
feat(dsplot): sample_template kitchen-sink showcase + render/show/embed three-mode contract

Adds research/dsplot/figures/sample_template.py — the canonical kitchen-sink
showcase exercising every panel kind, every plottable, the explicit
SuptitlePanel composition path, the Q2 cross-type overlay contracts
(Line-on-Heatmap in bin-space, TimeSeriesPanel twin-axis with Line),
and variable cell spans (1×1, 1×2, 1×3, 4×1). Four rows × 6-unit width,
show_cell_borders=True, zero hardcoded style literals.

Establishes the render() / show() / embed(target) three-mode invocation
contract as the canonical convention every figure module SHOULD implement
going forward. Contract documented in research/dsplot/figures/__init__.py
docstring; sample_template re-exported in __all__; existing figure_1 /
figure_2_4_1 re-exports untouched.

Notebook integration: src/subshader/dsp/dsp.ipynb gets a new cell calling
sample_template.show() for inline display. Rendered PNG lives at
assets/images/dsp/figures/sample_template/v1.png (iterate via v2.png,
v3.png, …).

Both worldviews verified — notebook sys.path-shim (worldview A) and
repo-root research.X absolute imports (worldview B).
```

## Deviations from Plan

### 1. Spotlight `mode="band"` does not exist (Rule 1 — known fallback)

The plan explicitly anticipated this and provided a fallback: "**CHECK the Spotlight signature** before authoring — if Spotlight only supports `mode='scatter'`, use that with `xy=(t_peak, y_peak)`." Confirmed: `research/dsplot/plottables/spotlight.py` accepts only `"rectangle"`, `"scatter"`, `"glow"`. Row 3 col 3 uses `mode="scatter"` anchored at the sine peak.

### 2. Inlined chirp builder instead of importing `utilities.build_waypoint_chirp` (deviation)

The plan allowed either `from utilities.dsp_helpers import build_waypoint_chirp` OR inline placeholder data. We inlined an iterated-log chirp helper (`_build_chirp_placeholder`) so the showcase has zero external data-helper dependencies — it's pure NumPy. The chirp is rendered in row 1 col 2 to give the TimeSeries + twin-axis Line a visually meaningful overlay (logarithmic frequency ramp from 20 Hz to 100 Hz).

### 3. Row 4 (4×4 bonus span) skipped per scope_brief

The plan called this out as bonus: "scope_brief says 4×4 span is bonus — skip in v1 to avoid awkward layout compromises. Note this skip in SUMMARY.md." Skipping noted here. Variable spans are still exercised via 1, 2, 3, 4 unit widths across rows 1–3.

### 4. No `from utilities import …` line in sample_template.py

Plan referenced `from utilities.dsp_helpers import build_waypoint_chirp` as an optional pattern. Since (2) eliminated the utility dependency, sample_template.py uses only `from .. import …` for dsplot-internal references — no `from utilities import …` either. This is fine: the `[[project-dsplot-import-worldviews]]` rule allows but does not require the utilities import.

## Known Visual Notes (for Human Verify)

- **InteractivePanel button overlap**: The Prev / Next button bar at the bottom of row 3 col 2 visually overlaps the "single-frame static export" subtitle. This is expected — InteractivePanel reserves `_base_bottom_pad = 0.18` (axes fractions) for controls, which the static PNG render captures as a strip below the axes. Not a regression; flagged so the human verifier doesn't mistake it for one.
- **All five rows visible with cell borders**: `show_cell_borders=True` produces gray rectangles around every cell — confirms the layout structure is intact and width-equality across rows holds (compose would have raised ValueError otherwise).

## Deferred / Open Questions

- `embed(target: matplotlib.figure.Figure)` and `embed(target: matplotlib.axes.Axes)` are NotImplementedError stubs in v1, reserved for future simpler single-panel figure modules. The kitchen-sink layout cannot currently re-host cleanly onto a caller-provided container; future single-panel figure modules will implement these paths and become the reference for that subset of the contract.
- 4×4 large bonus span skipped from v1 (see deviation 3). Reconsider for v2 if visual evidence suggests the showcase needs a panel with that footprint.
- Whether the v1 visual aesthetics need tuning across cells before the user iterates v2 is the human-verify call.

## Self-Check: PASSED

**Files exist:**
- FOUND: research/dsplot/figures/sample_template.py
- FOUND: research/dsplot/figures/__init__.py (modified)
- FOUND: src/subshader/dsp/dsp.ipynb (modified, 17 cells)
- FOUND: assets/images/dsp/figures/sample_template/v1.png (572,648 bytes)
- FOUND: .planning/quick/260521-tpz-master-sample-template-figure-exercising/260521-tpz-SUMMARY.md

**Smoke tests:**
- Worldview A: `A ok` printed
- Worldview B: `B ok` printed

**No executor commits:**
- `git log --oneline -3` head is unchanged from session start (a618f8a)
- All deliverables present as working-tree edits (`git status --short` shows the expected M/M/?? entries)

**Three-mode contract:**
- `sample_template.render()` returns absolute output path: confirmed
- `sample_template.show()` exists and is callable: confirmed
- `sample_template.embed(None)` returns a Figure: confirmed
- `sample_template.embed(Axes)` raises NotImplementedError: confirmed
- `sample_template.embed(Figure)` raises NotImplementedError: confirmed
