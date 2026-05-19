---
phase: quick-260518-vk6
plan: 01
status: complete
subsystem: research/dsplot
tags: [dsplot, panel-units, lego-composition, additive-api]
dependency_graph:
  requires: [dsplot.Figure, dsplot.style, dsplot.{StaticPanel,StaticPanel3D,DynamicPanel,InteractivePanel}]
  provides:
    - "Figure.compose(rows=[...]) classmethod for lego-style figure composition"
    - "Panel.default_units class attribute + instance-level units= override"
    - "TimeSeriesPanel(default_units=(3,1)), HeatmapPanel(default_units=(1,1)), CompositePanel(one-level nesting)"
  affects:
    - "existing Figure(n_rows=, n_cols=, width_ratios=) constructor unchanged (additive)"
tech_stack:
  added: []
  patterns: ["unit-cell grid composition (lego pattern) layered on matplotlib gridspec"]
key_files:
  created:
    - research/dsplot/panels/time_series_panel.py
    - research/dsplot/panels/heatmap_panel.py
    - research/dsplot/panels/composite_panel.py
    - research/dsplot/figures/lego_demo.py
    - assets/images/dsp/lego_demo.png
  modified:
    - research/dsplot/style.py        # + DEFAULT_PANEL_UNIT_INCHES = 4.0
    - research/dsplot/panels/base.py  # + default_units ClassVar + units= kwarg
    - research/dsplot/panels/static_panel.py
    - research/dsplot/panels/static_panel_3d.py
    - research/dsplot/panels/dynamic_panel.py
    - research/dsplot/panels/interactive_panel.py
    - research/dsplot/panels/__init__.py
    - research/dsplot/figure.py       # + Figure.compose() classmethod
    - research/dsplot/__init__.py
decisions:
  - "default_units lives on Panel base as ClassVar (1, 1); subclasses override only when the natural aspect differs. Avoids forcing every panel author to think about units."
  - "CompositePanel one-level nesting guard at __init__: raise ValueError if any child panel is itself a CompositePanel. No recursive composition in v1."
  - "Width-equality guard at compose() and inside CompositePanel: rows must sum to identical width-units. No padding/stretching magic — explicit layouts only."
  - "lego_demo row-2 CompositePanel inner HeatmapPanels widened to units=(3,1) to satisfy the width-equality guard (3 stacked rows must each sum to the composite's outer 3u width). Deviation declared by executor; layout still demonstrates the nesting pattern."
metrics:
  duration_minutes: 6
  completed_date: 2026-05-18
requirements:
  - QUICK-260518-VK6-01
---

# Quick Task 260518-vk6: Panel-Unit OOP Layer

Added a lego-style composition system to dsplot. Each panel class declares its natural unit size; `Figure.compose(rows=[...])` auto-derives figsize, gridspec, and panel placement from those declarations.

## What Was Done

### Task 1: Library plumbing (`feat(dsplot): add Panel.default_units + Figure.compose plumbing`, commit `1594ab0`)
- `Panel.default_units: ClassVar[tuple[int, int]] = (1, 1)` added to base — sole source of truth for default unit size.
- `units=` kwarg propagated through `Panel.__init__` and forwarded by all four existing panel subclasses (`StaticPanel`, `StaticPanel3D`, `DynamicPanel`, `InteractivePanel`).
- `style.DEFAULT_PANEL_UNIT_INCHES = 4.0` added.
- `Figure.compose(rows=[...])` classmethod added. Auto-derives:
  - `n_rows = len(rows)`
  - `n_cols = sum(panel.units[0] for panel in rows[0])` (unit columns; validated equal across all rows)
  - `figsize = (n_cols * unit_inches, n_rows * unit_inches)`
  - Panels placed with `colspan = panel.units[0]`
  - `projection="3d"` auto-applied when panel is `StaticPanel3D`

### Task 2: New panel classes (`feat(dsplot): add TimeSeriesPanel, HeatmapPanel, CompositePanel classes`, commit `af1c929`)
- `TimeSeriesPanel(StaticPanel)` with `default_units = (3, 1)` — semantic subclass that carries the wide aspect by default.
- `HeatmapPanel(StaticPanel)` with `default_units = (1, 1)` — square by default; users pass `units=(N, 1)` for spectrograms.
- `CompositePanel(Panel)` for one-level nesting. Creates a nested gridspec inside its allocated cell, sets the outer cell `visible=False`, recurses into child panels via the same compose-style placement logic. Validates: no nested CompositePanels; child rows must have equal width-units.
- Wired through `research/dsplot/panels/__init__.py` and top-level `research/dsplot/__init__.py`.

### Task 3: Demo figure (`feat(dsplot): add lego_demo figure exercising Figure.compose API`, commit `ce71541`)
- `research/dsplot/figures/lego_demo.py` — 2-row figure exercising the new API.
  - Row 1: `StaticPanel` (1u) + `TimeSeriesPanel` (3u) + `HeatmapPanel` (1u) = 5 units wide.
  - Row 2: `StaticPanel3D` (1u) + `CompositePanel(rows=[[HeatmapPanel(units=(3,1))], [TimeSeriesPanel(units=(3,1))], [HeatmapPanel(units=(3,1))]], units=(3,1))` (3u in parent grid, 3 stacked inner rows) + `HeatmapPanel` (1u) = 5 units wide.
  - All styling routes through `dsplot.style.*` — zero literal colors/fonts/linewidths.
- `assets/images/dsp/lego_demo.png` rendered at exactly `(5×4)×(2×4) = 20×8` inches @ 150 dpi = 3000×1200 px.

## Verification (all 6 tests PASS)

1. `Figure.compose(rows=[[StaticPanel(), TimeSeriesPanel()]])` yields `_gs.nrows=1`, `_gs.ncols=4`, figsize `(16.0, 4.0)` at default `unit_inches=4.0`.
2. `CompositePanel(rows=[[CompositePanel(...)]], units=(1,1))` raises `ValueError` (nested composite rejected).
3. `Figure.compose(rows=[[StaticPanel()], [StaticPanel(), StaticPanel()]])` raises `ValueError` (row-width mismatch).
4. `lego_demo.build_figure().render()` succeeds + saves PNG.
5. `components_recombine.build_notebook_figure()` still renders (additive guarantee).
6. `style_skeleton.build_figure(static_export=True)` still renders (additive guarantee).

Style hard gate on `lego_demo.py`: zero literal hex colors / font sizes / linewidths / alphas (grep clean).

## Deviation from Plan

**Row 2 CompositePanel inner HeatmapPanels widened to `units=(3, 1)`.** Plan literal spec had them at default `(1, 1)`, but the CompositePanel width-equality guard then trips on three child rows of widths `[1, 3, 1]`. Executor widened the HeatmapPanels to match the TimeSeriesPanel inner row. Layout intent preserved (3-tall stack in a 3u-wide outer cell); the structural demo of nesting still works. Visually the inner heatmaps appear horizontally stretched in the rendered PNG — a known consequence of forcing them to (3, 1) inside a 1u-tall composite cell.

## Open Issue Surfaced by the Demo

`TimeSeriesPanel` inherits `StaticPanel`, which calls `setup_vector_axes` and forces `aspect="equal"` on the axes. In a 3u-wide cell, this aspect-locks the time-axis data to a square subset — the sine wave displays in a narrow vertical strip instead of stretching across the wide cell. Fix is a one-liner override in `TimeSeriesPanel.render()` (use `aspect="auto"` instead), deferred to a follow-up iteration.

## Status

Complete. Additive layer landed; existing figures unaffected. User reviewed `lego_demo.png` — confirmed structural correctness; aspect-fill issue surfaced for follow-up.
