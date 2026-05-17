---
phase: 09-dsplot-library-architecture
plan: 04
type: summary
status: complete
verified_with_user: 2026-05-17
subsystem: dsplot (animated + interactive Panels)
tags: [dsplot, panel, dynamic-panel, interactive-panel, funcanimation, matplotlib-widgets, mixed-type-figure]
---

# 09-04 Summary — DynamicPanel + InteractivePanel + mixed Figures

## What shipped

**DynamicPanel** (`research/dsplot/panels/dynamic_panel.py`)
- `frames: list[list[Plottable]]` model; one Plottable list per frame.
- Renders via mpl `FuncAnimation`; `repeat=True` for auto-reset.
- Holds the FuncAnimation reference on `self._anim` so it survives `Figure.render()` return (mpl will garbage-collect orphaned animations).
- `save_gif(path, fps=...)` for non-interactive export.

**InteractivePanel** (`research/dsplot/panels/interactive_panel.py`)
- Same frame model as DynamicPanel.
- Controls implemented with `matplotlib.widgets` — NOT ipywidgets (D-07 amendment to original LOCKED choice; see 09-CONTEXT.md).
- Always-on: Prev / Next buttons centered just below the panel.
- Optional `slider=True`: `matplotlib.widgets.Slider` for direct frame index.
- Optional `checkbox=("label", callback)`: `matplotlib.widgets.CheckButtons` for toggling.
- Widgets render in `mpl_toolkits.axes_grid1.inset_locator.inset_axes` anchored to the panel's `transAxes` — controls live inside the figure canvas and stay column-aligned under their panel in multi-panel Figures.
- All widgets styled to the dark theme (SPINE_COLOR fill, NEUTRAL_COLOR text, PRIMARY_COLOR active state) so no white chrome leaks into the figure.

**Figure orchestrator updates** (`research/dsplot/figure.py`)
- New `Panel.requires_bottom_pad: float` class attribute. InteractivePanel sets it to `0.18`; `Figure.render()` takes the max across panels and calls `subplots_adjust(bottom=max + 0.05)`. Reserves room below the panel grid for controls without clipping.
- New Figure kwargs: `fill_width: bool = True`, `show_toolbar: bool = False`, `display_width: Optional[str] = None`. Drive ipympl-side display behavior — `canvas.layout.width = "100%"`, toolbar/header/footer hidden, optional explicit width.
- New `dsplot.apply_jupyter_dark()` helper: one-time CSS injection that paints `.cell-output-ipywidget-background`, `.jupyter-matplotlib`, `.jupyter-widgets.widget-container`, and `.jp-OutputArea-output` to `style.BG_COLOR`. Called automatically by `Figure.render()` under an ipympl backend; idempotent within a session. Re-exported from the `dsplot` top-level for manual invocation.

**Smoke-test notebook** (`research/dsplot/notebooks/09_04_smoke_test.ipynb`)
- 5 cells, each pinned to one LOCKED checkpoint criterion:
  1. DynamicPanel alone — rotates and loops with auto-reset.
  2. InteractivePanel alone — prev/next buttons centered below the panel.
  3. Mixed Static + Dynamic Figure.
  4. Mixed Static + Interactive Figure.
  5. Mixed Dynamic + Interactive Figure (the LOCKED primary mixed-type goal).

## User verification (2026-05-17)

The 4 LOCKED checkpoint confirmations were performed interactively in VS Code Jupyter with `%matplotlib widget`:

1. **Jupyter stack installed** — `jupyterlab 4.5.7`, `ipywidgets 8.1.8`, `ipympl 0.10.0`, `notebook 7.5.6` installed into the project venv. (ipywidgets ultimately not needed by the library after D-07; kept for ipympl runtime.)
2. **`%matplotlib widget` works** — VS Code Jupyter rendered the dark-theme figures via the ipympl canvas widget.
3. **Mixed-type composition** — verified visually for Static+Dynamic (Smoke 3), Static+Interactive (Smoke 4), and **Dynamic+Interactive** (Smoke 5 — the LOCKED primary goal). All three composed without conflict; the dynamic side animates while the interactive side responds to clicks on its prev/next buttons.
4. **Same-type-only fallback (non-goal #6) NOT triggered** — full mixed-type primary goal is met.

User feedback during verification drove three follow-up changes that all landed in this plan:
- Drop the slider from the default smoke-test layout (kept as opt-in via `slider=True`).
- Pivot from ipywidgets to `matplotlib.widgets` for visual grouping + dark-theme consistency (D-07 amendment).
- Add Jupyter display styling (`fill_width`, `show_toolbar`, CSS injection) so the figure fills the cell and the cell-output container is dark, no white chrome.

## Known limitations (deferred, not blocking)

- **CSS scaling, not auto-rerender.** When the cell container width changes (VS Code zoom, panel toggle), the figure widget scales via CSS — the underlying mpl figure does not re-render at the new pixel size. Acceptable per user; deferred as a potential follow-up (would require a JS `ResizeObserver` + Comm channel back to Python).
- **`requires_bottom_pad` is figure-wide**, not per-row. In an N×M layout with InteractivePanels only in row 0, the bottom margin still applies to the whole figure. Multi-row layouts with InteractivePanels nested below other rows would need a more sophisticated reservation strategy. Out of scope for 09-04.

## Tests

`research/tests/dsplot/test_interactive_panel.py` rewritten for the mpl-widgets pivot. Full dsplot suite: **109 passed, 4 warnings** (no skips — previously had 3 skipped under ipywidgets-importorskip gates that are no longer needed).

| Test file | Tests |
|---|---:|
| `test_plottable_construction.py` | 30 |
| `test_vector_plottables.py` | 17 |
| `test_spotlight_dropline.py` | 10 |
| `test_dynamic_panel.py` | 9 |
| `test_panel_composition.py` | 9 |
| `test_figure_orchestrator.py` | 9 |
| `test_time_series_heatmap.py` | 8 |
| `test_interactive_panel.py` | 8 |
| `test_annotation.py` | 6 |
| `test_mixed_figure.py` | 3 |
| **Total** | **109** |

D-01 isolation grep `grep -rIn "subshader" research/dsplot/ --exclude-dir=figures` returns zero matches.

## Files touched in this plan

- `research/dsplot/panels/dynamic_panel.py` — DynamicPanel
- `research/dsplot/panels/interactive_panel.py` — InteractivePanel (mpl-widgets pivot)
- `research/dsplot/panels/base.py` — added `requires_bottom_pad` class attribute
- `research/dsplot/panels/__init__.py` — DynamicPanel + InteractivePanel exports
- `research/dsplot/figure.py` — bottom-pad reservation + ipympl display styling + `apply_jupyter_dark`
- `research/dsplot/__init__.py` — exports `apply_jupyter_dark`
- `research/tests/dsplot/test_dynamic_panel.py` — DynamicPanel tests
- `research/tests/dsplot/test_interactive_panel.py` — InteractivePanel tests (mpl-widgets pivot)
- `research/tests/dsplot/test_mixed_figure.py` — Static+Dynamic, Static+Interactive
- `research/dsplot/notebooks/09_04_smoke_test.ipynb` — 5-smoke verification notebook
- `.planning/phases/09-dsplot-library-architecture/09-CONTEXT.md` — D-07 amendment recorded
