---
phase: quick-260521-tpz
plan: 01
type: execute
wave: 1
depends_on: []
subsystem: research/dsplot/figures
tags: [dsplot, sample_template, figures, suptitle_panel, showcase, notebook]
autonomous: false
requirements:
  - TPZ-G1-sample-template-module
  - TPZ-G2-three-mode-contract
  - TPZ-G3-render-and-verify
  - TPZ-G4-notebook-integration
files_modified:
  - research/dsplot/figures/sample_template.py
  - research/dsplot/figures/__init__.py
  - src/subshader/dsp/dsp.ipynb
  - assets/images/dsp/figures/sample_template/v1.png
must_haves:
  truths:
    - "sample_template.py renders a kitchen-sink PNG that demonstrates every panel kind, every plottable, the new SuptitlePanel composition path, and the Q2 cross-type overlay contracts in a single figure"
    - "The figure derives its visual style from the v52 lego_demo language (unitary PAD knob, gutter, margins, fonts, colors) via dsplot.style — zero hardcoded style literals in the new module"
    - "Every dsplot figure-module convention going forward is documented inside research/dsplot/figures/__init__.py: render() / show() / embed(target) with the signatures used by sample_template"
    - "Both worldview smoke tests pass against the new module (sys.path-shim notebook style AND repo-root research.X style)"
    - "An absolute PNG path is surfaced in SUMMARY.md for the user to eyeball"
    - "dsp.ipynb has (or has a SUMMARY.md snippet for) a sample_template.show() cell"
    - "Zero git commits made by the executor — all changes left as working-tree edits for user review per feedback-no-auto-commits"
  artifacts:
    - path: "research/dsplot/figures/sample_template.py"
      provides: "Kitchen-sink showcase module with render() / show() / embed(target) entrypoints + private _build_figure() helper"
      contains: "def render(", "def show(", "def embed(", "def _build_figure(", "SuptitlePanel("
    - path: "research/dsplot/figures/__init__.py"
      provides: "One-paragraph docstring documenting the render/show/embed three-mode contract that every figure module SHOULD follow"
      contains: "render(", "show(", "embed("
    - path: "assets/images/dsp/figures/sample_template/v1.png"
      provides: "Rendered PNG of the showcase figure for visual verification"
    - path: ".planning/quick/260521-tpz-master-sample-template-figure-exercising/260521-tpz-SUMMARY.md"
      provides: "Self-check, smoke-test output, absolute PNG path, ipynb integration notes / paste snippet"
  key_links:
    - from: "research/dsplot/figures/sample_template.py"
      to: "research/dsplot/__init__.py"
      via: "from .. import SuptitlePanel, StaticPanel, StaticPanel3D, DynamicPanel, InteractivePanel, TimeSeriesPanel, HeatmapPanel, CompositePanel, ... , style"
      pattern: "from \\.\\. import"
    - from: "research/dsplot/figures/sample_template.py"
      to: "research/utilities/dsp_helpers.py"
      via: "from utilities.dsp_helpers import build_waypoint_chirp (or inline np.sin for placeholder data)"
      pattern: "from utilities"
    - from: "src/subshader/dsp/dsp.ipynb"
      to: "research/dsplot/figures/sample_template.py"
      via: "from dsplot.figures import sample_template; sample_template.show()"
      pattern: "sample_template"
---

<objective>
Final deliverable of the 3-quick dsplot batch (Q1 cleanup + Q2 SuptitlePanel/overlays → Q3 master template). Author a brand-new `research/dsplot/figures/sample_template.py` that is the most demonstrative possible kitchen-sink showcase of the dsplot framework: every panel kind, every plottable, the new explicit-SuptitlePanel composition path, the Q2 cross-type overlay contracts (Line-on-Heatmap in bin-space, TimeSeriesPanel twin-axis with Line), and variable cell spans (1×1, 1×2, 2×2, 1×3). Establish the `render()` / `show()` / `embed(target)` three-mode invocation contract and document it as the canonical convention every future figure module should follow.

Purpose: Give the user a single iterable PNG (`assets/images/dsp/figures/sample_template/v1.png` → `v2.png` → `vN.png`) that visualises every framework capability at once. Going forward, when the visual style language evolves, this template is the first thing they re-render to see the impact across the whole API surface. Also locks in the three-mode contract so subshader tests/benchmarks (`embed`) and dsp.ipynb cells (`show`) get a consistent interface across every figure module.

Output:
- NEW `research/dsplot/figures/sample_template.py` — composed via `Figure.compose(rows=[[SuptitlePanel(...)], ...])` with variable spans + every panel + every plottable + cross-type overlays.
- MODIFIED `research/dsplot/figures/__init__.py` — adds the convention paragraph (no behavioral change to existing re-exports).
- MODIFIED `src/subshader/dsp/dsp.ipynb` — adds one cell calling `sample_template.show()` (or SUMMARY.md fallback snippet if JSON edit is risky).
- NEW PNG `assets/images/dsp/figures/sample_template/v1.png`.
- NEW SUMMARY.md surfacing the PNG path + smoke-test output.
</objective>

<execution_context>
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/workflows/execute-plan.md
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@.planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md
@research/dsplot/__init__.py
@research/dsplot/figures/__init__.py
@research/dsplot/figures/lego_demo.py
@research/dsplot/figures/figure_1.py
@research/dsplot/figures/style_skeleton.py
@research/dsplot/panels/suptitle_panel.py
@research/dsplot/style.py
@research/utilities/dsp_helpers.py

<interfaces>
<!-- Extracted from current codebase so the executor does NOT need to re-explore. Use these directly. -->

# Top-level dsplot exports (research/dsplot/__init__.py)
# All importable inside research/dsplot/figures/*.py via `from .. import …`.
Figure, apply_jupyter_dark
Panel, StaticPanel, StaticPanel3D, DynamicPanel, InteractivePanel,
TimeSeriesPanel, HeatmapPanel, CompositePanel, TextPanel, SuptitlePanel
Vector, VectorComponents, Annotation, TimeSeries, Heatmap, Line,
Spotlight, Stem, Dropline
style, freq_axis

# Figure.compose signature (research/dsplot/figure.py:217)
Figure.compose(
    *,
    rows: List[List[Panel]],
    suptitle: Optional[str] = None,
    suptitle_fontsize: Optional[float] = None,
    unit_inches: Optional[float] = None,
    unit_height_inches: Optional[float] = None,
    dpi: Optional[int] = None,
    hspace: Optional[float] = None,
    wspace: Optional[float] = None,
    debug_guides: bool = False,
    show_cell_borders: bool = False,         # MUST pass True per scope_brief
    top_reserve_inches: Optional[float] = None,
    bottom_reserve_inches: Optional[float] = None,
    figure_number: Optional[str] = None,
    figure_caption: Optional[str] = None,
) -> Figure

# Figure (post-compose) lifecycle:
fig.render()          # builds + draws (call before show/savefig)
fig.savefig(path, **kwargs) -> str   # saves at exactly figsize × dpi
fig._mpl_fig          # underlying matplotlib Figure (for show() chrome suppression)

# SuptitlePanel signature (research/dsplot/panels/suptitle_panel.py:37)
SuptitlePanel(
    text: str,
    *,
    units: Optional[Tuple[int, int]] = None,   # MUST set units=(N, 1) where N == row width
    font_size: Optional[float] = None,         # defaults to style.SUPTITLE_FONT_SIZE (32)
    color: Optional[str] = None,               # defaults to style.SUPTITLE_COLOR ("#888888")
    fontweight: Optional[str] = None,          # defaults to style.SUPTITLE_WEIGHT ("bold")
    auto_shrink: bool = False,
)

# Width-equality invariant — Figure.compose validates that every row's
# sum(p.units[0] for p in row) is identical. Mismatch raises ValueError.

# Style constants the showcase MUST respect (no hardcoded style literals):
style.DEFAULT_PANEL_UNIT_INCHES  # 4.0
style.DEFAULT_MARGIN_INCHES      # 1.0 (= DEFAULT_PAD_INCHES — the v52 unitary PAD)
style.DEFAULT_GUTTER_INCHES      # 2.0 (= 2 * PAD)
style.DEFAULT_COLUMN_GUTTER_INCHES  # 2.0 (= 2 * PAD)
style.DEFAULT_DPI                # 150 (production default)
style.SUPTITLE_FONT_SIZE         # 32
style.PRIMARY_COLOR, SECONDARY_COLOR, NEUTRAL_COLOR, HIGHLIGHT_COLOR,
style.TICK_LABEL_COLOR, BG_COLOR, DROPLINE_COLOR
style.DEFAULT_VECTOR_LINEWIDTH, DEFAULT_VECTOR_BOLD_LINEWIDTH,
style.DEFAULT_DROPLINE_LINEWIDTH, INST_FREQ_LINEWIDTH, INST_FREQ_COLOR,
style.INST_FREQ_ALPHA

# Panel default_units (Panel base + subclasses):
Panel              # (1, 1)
StaticPanel        # (1, 1)
StaticPanel3D      # (1, 1)
HeatmapPanel       # (1, 1)
DynamicPanel       # (1, 1)
InteractivePanel   # (1, 1)
TextPanel          # (1, 1)
SuptitlePanel      # inherits from TextPanel → (1, 1) — override with units=(N, 1)
TimeSeriesPanel    # (3, 1)
CompositePanel     # caller-provided, no default; pass units=(W, H)

# DynamicPanel signature (research/dsplot/panels/dynamic_panel.py:45)
DynamicPanel(
    *,
    units=None,
    frames: Optional[List[Frame]] = None,            # each Frame = List[Plottable]
    frame_fn: Optional[Callable[[int], Frame]] = None,
    num_frames: Optional[int] = None,
    interval_ms: int = 250,
    repeat: bool = True,
    base_plottables: Optional[List[Plottable]] = None,
    title=None, subtitle=None, caption=None, lim=None,
    axis_style="line", axis_labels=False, show_border=True,
    show_ticks=False, show_grid=False, tick_positions=None,
)
# For DynamicPanel in a STATIC PNG render, just pass `frames=[<single Frame>]`
# so the animation displays its first (and only) frame — savefig captures that.

# InteractivePanel signature (research/dsplot/panels/interactive_panel.py:47)
InteractivePanel(
    *,
    units=None,
    frames: List[Frame],                              # at least 1 frame
    base_plottables=None, slider=False, checkbox=None,
    title=None, subtitle=None, caption=None, lim=None,
    axis_style="line", axis_labels=False, show_border=True,
)
# For static PNG render, also pass `frames=[<single Frame>]`. The control bar
# (prev/next buttons) reserves _base_bottom_pad=0.18 in axes fractions; this is
# fine — it just leaves a strip below the axes in the static PNG.

# Heatmap signature (research/dsplot/plottables/heatmap.py:38) — Q2 added alpha=
Heatmap(
    data,
    *,
    duration_s=None, freqs=None, log_freq=False, tick_freqs=(20,200,2000,20000),
    cmap=None, vmin=0.0, vmax=None, vmax_percentile=None,
    alpha: float = 1.0,                               # Q2 deliverable
    origin="lower", aspect="auto",
    extent: tuple[float,float,float,float] | None = None,
    zorder: int = 1,
)

# Line signature (research/dsplot/plottables/line.py:31) — generic 1D line
Line(
    x, y,
    *,
    color=None, linewidth=None, alpha=1.0, linestyle="-",
    label=None, zorder=3,
)

# TimeSeriesPanel.twin_y / add_twin API (added in quick 260519-a1p)
TimeSeriesPanel(
    units=None, x_label=None, y_label=None, y_label_side="left",
    xticks=None, yticks=None,
    twin_y: bool = False, twin_y_side="right" | "left",
    twin_y_label=None, twin_yticks=None, twin_ytick_labels=None, twin_ylim=None,
    ...,
)
panel.add_twin(Line(...))    # plottable goes onto the twin axis

# HeatmapPanel Line-overlay contract (Q2 docstring): caller must
# pre-transform Hz → bin index via np.interp before constructing the Line.
# Then `heatmap_panel.add(Line(x_seconds, y_bins, ...))` lands on the primary
# heatmap axis (NO twin y-axis on HeatmapPanel).

# dsp_helpers signal builders (research/utilities/dsp_helpers.py)
build_waypoint_chirp(sr, duration_s, waypoints, clip_to_waypoints=False)
    # returns (signal: np.ndarray, inst_freq: np.ndarray, t: np.ndarray)
# Sufficient for the showcase. Alternative: inline `t = np.arange(...)/sr;
# sig = np.sin(2*pi*f*t)` — placeholder data has no DSP storytelling rule.

# Existing figures/__init__.py re-exports — leave them intact, only ADD the
# new convention paragraph + (optionally) a re-export for sample_template:
from .components_recombine import show as figure_2_4_1
from .figure_1 import show_hero as figure_1

</interfaces>

<conventions>
- dsplot import worldview rule [[project-dsplot-import-worldviews]]: inside
  `research/dsplot/figures/sample_template.py` use ONLY:
    - `from .. import Figure, SuptitlePanel, StaticPanel, ..., style`
    - `from utilities.dsp_helpers import build_waypoint_chirp` (if used)
  NEVER `from research.X import …` — that breaks worldview A (notebook).
- Style language: derive every visual knob from `style.*` — no hardcoded
  font sizes, line widths, colors, margins, gutters, or DPIs in the new module.
  Data constants (signal frequencies, vector tips, grid sizes) are allowed
  as module-level literals (see lego_demo's SAMPLE_RATE, A2, A3 pattern).
- Voice rule (CLAUDE.md): descriptive names, no comment litter, structure
  over docs. Module docstring + per-function docstring is fine; avoid
  step-by-step running commentary.
- Per [[feedback-no-auto-commits]]: ZERO git commits by the executor. Surface
  proposed commit message + file list in SUMMARY.md instead.
</conventions>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Author research/dsplot/figures/sample_template.py — kitchen-sink showcase + three-mode contract</name>
  <files>research/dsplot/figures/sample_template.py</files>
  <action>
Create the new module from scratch. NO git commits — per feedback-no-auto-commits, leave all changes as working-tree edits.

**Layout — pick row width N = 6 units (matches a 1+2+2+1 or 1+3+2 row composition with headroom for variable spans). All rows must sum to 6.**

Compose via `Figure.compose(rows=[[SuptitlePanel(...)], [<row 1>], [<row 2>], [<row 3>], [<row 4>]])` — five rows total: one SuptitlePanel row plus four data rows. SuptitlePanel goes in row 0 with `units=(6, 1)` so it spans the full grid width. Pass `show_cell_borders=True` to Figure.compose (per scope_brief — makes the layout structure visually explicit).

**Row 0 (suptitle row):**
- `SuptitlePanel("Sample Template — dsplot Showcase", units=(6, 1))`

**Row 1 (variable spans 1×1 + 1×2 + 1×3 = 6 units; covers StaticPanel, TimeSeriesPanel-with-twin, HeatmapPanel-with-Line-overlay):**
- `StaticPanel(title="2D Vector", units=(1, 1), axis_labels=True, show_ticks=True, show_grid=True)` — add `VectorComponents((3.0, 4.0), from_origin=True, component_color=style.DROPLINE_COLOR, dropline_color=style.DROPLINE_COLOR, label_x="aₓ", label_y="aᵧ")` and `Vector((3.0, 4.0), color=style.PRIMARY_COLOR, linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH, label="a", zorder=4)`. Exercises **StaticPanel + Vector + VectorComponents**.
- `TimeSeriesPanel(units=(2, 1), title="TimeSeries + Twin Line", x_label="t (s)", y_label="amp", xticks=[0.0, 0.5, 1.0], yticks=[-1.0, 0.0, 1.0], twin_y=True, twin_y_side="right", twin_y_label="inst f (Hz)", twin_yticks=[20.0, 100.0], twin_ylim=(20.0, 100.0))` — add `TimeSeries(sig, sr, color=style.NEUTRAL_COLOR, alpha=0.75)` (the chirp signal from `_build_chirp_placeholder`) and on the twin axis `panel.add_twin(Line(t, inst_f, color=style.PRIMARY_COLOR, linewidth=style.INST_FREQ_LINEWIDTH + 1.0, alpha=style.INST_FREQ_ALPHA))`. Exercises **TimeSeriesPanel + TimeSeries + Line (twin-axis)** — rhymes with figure_1 row 1 per scope_brief.
- `HeatmapPanel(units=(3, 1), title="Heatmap + Line overlay (bin-space)", x_label="t (s)", y_label="bin")` — add a `Heatmap(field_2d, extent=(0.0, 1.0, 0.0, float(GRID_N)), aspect="auto", alpha=0.95)`. Then add a `Line` on the primary axis in bin-space per the Q2 contract: pre-transform a Hz-domain frequency track to bin indices via `np.interp(track_hz, np.linspace(0, NYQ, GRID_N), np.arange(GRID_N))` and pass `Line(t, bin_indices, color=style.HIGHLIGHT_COLOR, linewidth=style.INST_FREQ_LINEWIDTH, alpha=0.9)`. Exercises **HeatmapPanel + Heatmap + Line (heatmap overlay)**.

**Row 2 (variable spans 1×1 + 2×2 footprint via a CompositePanel + 1×1 = 6 units; covers StaticPanel3D, CompositePanel nesting, dropline/stem/spotlight/annotation):**
- `StaticPanel3D(units=(1, 1), title="3D Vector", lim_3d=5.0, spine_extension=1.7)` — add a 3D `Vector((3.0, 3.0, 2.0), color=style.PRIMARY_COLOR, linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH, show_arrowhead=True, zorder=5)` plus three dashed projection `Vector`s with `show_tip=False` (copy the pattern from lego_demo `panel_3d` component staircase). Exercises **StaticPanel3D + 3D Vectors**.
- `CompositePanel(units=(4, 1), title="Composite (Stem + Annotation/Spotlight + Dropline)", rows=[[<inner row>]])` where inner row is exactly ONE TimeSeriesPanel cell with units=(4, 1), inside a single composite row. Inside that inner TimeSeriesPanel:
  - Build a discretely-sampled signal `t_stem = np.linspace(0, 1, 32, endpoint=False); y_stem = np.sin(2*np.pi*3.0*t_stem)`.
  - Add `Stem(t_stem, y_stem, color=style.PRIMARY_COLOR)`.
  - Pick the peak sample (argmax) for highlight: `peak_t = float(t_stem[idx])`, `peak_y = float(y_stem[idx])`.
  - Add `Dropline(start=(peak_t, peak_y), end=(peak_t, 0.0))`.
  - Add `Spotlight(mode="scatter", xy=(peak_t, peak_y))`.
  - Add `Annotation("peak", xy=(peak_t + 0.05, peak_y + 0.15), arrow_to=(peak_t, peak_y), color=style.HIGHLIGHT_COLOR, fontsize=style.DEFAULT_ANNOTATION_FONT_SIZE, fontweight="bold")`.
  This single cell covers **CompositePanel + Stem + Dropline + Spotlight + Annotation** in one place (Annotation + Spotlight sharing a cell is explicitly allowed by scope_brief).
- `HeatmapPanel(units=(1, 1), title="Heatmap 2", x_label="x", y_label="y", xticks=[-1.0, 0.0, 1.0], yticks=[-1.0, 0.0, 1.0])` with `Heatmap(gaussian_field, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal")`. Plus exercise **plottable: Heatmap** in a second context.

**Row 3 (DynamicPanel + InteractivePanel + StaticPanel-spotlight-only filler; spans 2×1 + 2×1 + 2×1 = 6 units):**
- `DynamicPanel(units=(2, 1), title="DynamicPanel", subtitle="single-frame static export", frames=[[TimeSeries(sig, sr, color=style.PRIMARY_COLOR)]], interval_ms=400, axis_style="line", show_border=True)`. Single-frame `frames=[<one Frame>]` is the static-PNG idiom from style_skeleton — savefig captures the first/only frame. Exercises **DynamicPanel**.
- `InteractivePanel(units=(2, 1), title="InteractivePanel", subtitle="single-frame static export", frames=[[Heatmap(field_2d_alt, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal", vmax_percentile=100.0)]], slider=True, show_border=True)`. Same single-frame idiom. Exercises **InteractivePanel**.
- `StaticPanel(units=(2, 1), title="TimeSeries + Spotlight", axis_style="line", show_border=True)` — add `TimeSeries(sin_signal, sr, color=style.PRIMARY_COLOR)` plus a `Spotlight(mode="band", xy=(0.25, 0.0), width=0.10)` if `mode="band"` is supported; if Spotlight only supports `mode="scatter"`, use that with `xy=(t_peak, y_peak)`. **CHECK the Spotlight signature** before authoring — `grep -n "def __init__\|mode" research/dsplot/plottables/spotlight.py`. Exercises a second **Spotlight** usage. (This is the deliberate redundancy required to hit "every plottable" coverage in a single row.)

**Row 4 (optional bonus): scope_brief says 4×4 span is bonus — skip in v1 to avoid awkward layout compromises. Note this skip in SUMMARY.md.**

**Coverage audit table (include as a docstring section in the module — keeps the showcase self-documenting):**

| Panel kind        | Where in showcase                                                 |
|-------------------|-------------------------------------------------------------------|
| StaticPanel       | row 1 col 1 (2D Vector); row 3 col 3 (TimeSeries + Spotlight)     |
| StaticPanel3D     | row 2 col 1 (3D Vector)                                           |
| TimeSeriesPanel   | row 1 col 2 (twin); inside composite row 2 col 2                  |
| HeatmapPanel      | row 1 col 3 (Line overlay); row 2 col 3 (gaussian)                |
| CompositePanel    | row 2 col 2                                                       |
| DynamicPanel      | row 3 col 1                                                       |
| InteractivePanel  | row 3 col 2                                                       |
| TextPanel         | via SuptitlePanel (subclass of TextPanel) in row 0                |
| SuptitlePanel     | row 0 (explicit composition path — Q2 deliverable)                |

| Plottable          | Where in showcase                                                |
|--------------------|------------------------------------------------------------------|
| TimeSeries         | row 1 col 2; row 3 col 1 (dynamic frame); row 3 col 3            |
| Heatmap            | row 1 col 3; row 2 col 3; row 3 col 2 (interactive frame)        |
| Line               | row 1 col 2 (twin); row 1 col 3 (heatmap overlay, bin-space)     |
| Vector             | row 1 col 1 (2D); row 2 col 1 (3D)                                |
| VectorComponents   | row 1 col 1                                                       |
| Annotation         | inside composite row 2 col 2                                      |
| Dropline           | inside composite row 2 col 2                                      |
| Spotlight          | inside composite row 2 col 2; row 3 col 3                         |
| Stem               | inside composite row 2 col 2                                      |

**Three-mode contract — implement EXACTLY per scope_brief.**

```python
def _build_figure(unit_inches=None, dpi=None) -> Figure:
    # builds the composed Figure with optional sizing/DPI overrides
    ...

def render(
    output_dir: str = "assets/images/dsp/figures/sample_template",
    output_filename: str = "v1.png",
) -> str:
    """Build, render, and save the figure. Returns absolute output path."""
    fig = _build_figure()         # production DPI (style.DEFAULT_DPI=150), unit_inches=style.DEFAULT_PANEL_UNIT_INCHES=4.0
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)

def show() -> Figure:
    """Notebook-tuned rendering for dsp.ipynb inline display."""
    import matplotlib.pyplot as plt
    fig = _build_figure(unit_inches=2.5, dpi=60)
    fig.render()
    canvas = fig._mpl_fig.canvas
    for attr in ("header_visible", "toolbar_visible", "footer_visible"):
        try:
            setattr(canvas, attr, False)
        except Exception:
            pass
    try:
        canvas.manager.set_window_title("")
    except Exception:
        pass
    plt.show()
    return fig

def embed(target: object | None = None) -> Figure:
    """Drop into a caller-provided matplotlib container.

    v1 behaviour:
      - target is None: behave like show() WITHOUT chrome suppression
        (returns the Figure for the caller to display however they like).
      - target is a matplotlib.figure.Figure: NotImplementedError with a
        TODO message — sample_template is kitchen-sink; re-hosting onto a
        caller-provided Figure is reserved for v2.
      - target is a matplotlib.axes.Axes: NotImplementedError with a TODO —
        Axes-targeting is reserved for simpler single-panel figure modules.
    """
    import matplotlib.axes
    import matplotlib.figure as mpl_figure
    if target is None:
        fig = _build_figure(unit_inches=2.5, dpi=60)
        fig.render()
        return fig
    if isinstance(target, mpl_figure.Figure):
        raise NotImplementedError(
            "sample_template.embed(target: Figure) is reserved for v2 — "
            "the kitchen-sink layout cannot currently re-host onto a caller "
            "Figure cleanly. Use render() for static PNGs or show() for notebook display."
        )
    if isinstance(target, matplotlib.axes.Axes):
        raise NotImplementedError(
            "sample_template.embed(target: Axes) is not supported — Axes-targeting "
            "is reserved for simpler single-panel figure modules. Use render() or show()."
        )
    raise TypeError(
        f"sample_template.embed(target=...) expects None, matplotlib Figure, or matplotlib Axes; got {type(target).__name__}"
    )
```

**Module skeleton (no leading commentary or running narration in code per CLAUDE.md voice rule):**

```python
"""sample_template — kitchen-sink dsplot showcase.

Exercises every Panel kind and every Plottable in one Figure. Visual style
derives from the v52 lego_demo language via dsplot.style — no hardcoded
style literals.

Three-mode contract (the canonical convention every figure module should
implement):
    render(output_dir, output_filename) -> str   # production PNG
    show() -> Figure                              # notebook inline display
    embed(target) -> Figure                       # caller-provided container

Iterate the visual style by bumping output_filename: v1.png -> v2.png -> vN.png.
"""
from __future__ import annotations

import os
import numpy as np

from .. import (
    Annotation, CompositePanel, Dropline, DynamicPanel, Figure, Heatmap,
    HeatmapPanel, InteractivePanel, Line, Spotlight, StaticPanel,
    StaticPanel3D, Stem, SuptitlePanel, TimeSeries, TimeSeriesPanel,
    Vector, VectorComponents, style,
)

# Data constants (not style).
SAMPLE_RATE = 1000.0
DURATION = 1.0
SIGNAL_FREQ = 3.0
GRID_N = 64
NYQ = SAMPLE_RATE / 2.0
A2 = (3.0, 4.0)
A3 = (3.0, 3.0, 2.0)
LIM_3D = 5.0
STEM_N = 32

# (private builders: _build_sin, _build_chirp_placeholder, _build_gaussian,
# _build_field_with_freq_track) — implement as small helpers returning
# np.ndarray data only, no Plottables.

def _build_figure(unit_inches=None, dpi=None) -> Figure:
    ...

def render(...) -> str: ...
def show() -> Figure: ...
def embed(target=None) -> Figure: ...
```

**Style discipline check before saving:** grep your own draft for any of these literals — they should NOT appear (use the `style.*` constant instead):
- `fontsize=\d`, `linewidth=\d`, hex colors `#[0-9a-f]{6}`, `cmap="..."`, explicit `dpi=\d` outside `show()`/`render()` defaults.

Verify the worldview-A and worldview-B smoke tests both pass after creating the file.
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader &amp;&amp; source venv/bin/activate &amp;&amp; python -c "
import sys
sys.path.insert(0, 'research')
from dsplot.figures import sample_template
# Build but don't save — confirm composition succeeds end-to-end
fig = sample_template._build_figure()
fig.render()
print('compose ok; n_rows=', len(fig._gs.get_geometry() if hasattr(fig._gs, 'get_geometry') else (5, 6)))
# Confirm the three-mode contract exists
assert callable(sample_template.render), 'render() missing'
assert callable(sample_template.show), 'show() missing'
assert callable(sample_template.embed), 'embed() missing'
# Confirm embed(None) returns a Figure
import matplotlib
matplotlib.use('Agg')
fig2 = sample_template.embed(None)
print('embed(None) ok, type=', type(fig2).__name__)
# Confirm embed(target: Axes) raises NotImplementedError
import matplotlib.pyplot as plt
test_fig, test_ax = plt.subplots()
try:
    sample_template.embed(test_ax)
    print('FAIL: embed(Axes) should have raised')
    sys.exit(1)
except NotImplementedError:
    print('embed(Axes) NotImplementedError ok')
print('Task 1 OK')
"</automated>
  </verify>
  <done>
- `research/dsplot/figures/sample_template.py` exists.
- Module exposes `_build_figure`, `render`, `show`, `embed`.
- `Figure.compose(rows=[[SuptitlePanel(...)], [row1], [row2], [row3]])` with row width N=6 succeeds (no ValueError).
- All four panel kinds (Static / Static3D / Dynamic / Interactive) plus TimeSeriesPanel + HeatmapPanel + CompositePanel + SuptitlePanel are instantiated.
- All plottables (TimeSeries / Heatmap / Line / Vector / VectorComponents / Annotation / Dropline / Spotlight / Stem) appear at least once.
- TimeSeriesPanel uses `twin_y=True` + `add_twin(Line(...))` at least once.
- HeatmapPanel hosts a `Line` overlay in bin-space at least once.
- `show_cell_borders=True` in the Figure.compose call.
- Worldview A smoke test passes ("Task 1 OK").
- NO git commits made.
  </done>
</task>

<task type="auto">
  <name>Task 2: Document the three-mode contract in research/dsplot/figures/__init__.py</name>
  <files>research/dsplot/figures/__init__.py</files>
  <action>
Modify the existing `research/dsplot/figures/__init__.py` to add a one-paragraph docstring section documenting the `render()` / `show()` / `embed(target)` three-mode contract — the canonical convention every figure module SHOULD implement going forward. NO git commits.

**DO NOT retrofit figure_1.py / motivator.py / components_recombine.py / lego_demo.py / style_skeleton.py.** Per scope_brief: just establish + document.

**Preserve the existing module docstring + re-exports** (`figure_1`, `figure_2_4_1`). Append the convention paragraph to the existing docstring, keep `__all__` intact, and optionally (NOT required) add `from .sample_template import show as sample_template` to re-export for ipynb consumption. Decide: re-exporting matches the existing `figure_1` / `figure_2_4_1` pattern → DO re-export. Update `__all__` to add `"sample_template"`.

**New docstring section** (append after the existing "Notebook re-exports" paragraph):

```
Three-mode invocation contract (convention for every figure module)
====================================================================

Every figure module in this subpackage SHOULD expose three entrypoints
with identical signatures:

    render(output_dir: str, output_filename: str) -> str
        Build, render, and save the figure at production DPI / unit_inches.
        Returns absolute output path. Used by the __main__ batch dispatcher
        and by anything that wants a static PNG on disk.

    show() -> Figure
        Notebook-tuned rendering for dsp.ipynb inline display. Smaller DPI
        (~60) and unit_inches (~2.5) so the figure fits inside a Jupyter
        cell. Suppresses ipympl widget chrome on the returned Figure before
        plt.show() so the canvas reads as pure figure content.

    embed(target: object | None = None) -> Figure
        Drop the figure into a caller-provided matplotlib container.
        `target=None` behaves like show() without chrome suppression
        (subshader tests / benchmarks own their own display loop).
        `target: matplotlib.axes.Axes` re-hosts a single panel into the
        caller's Axes — only supported by single-panel figure modules.
        `target: matplotlib.figure.Figure` re-hosts the whole layout —
        supported only where the layout can adapt cleanly.

Implementation pattern: a private ``_build_figure(unit_inches=None,
dpi=None) -> Figure`` helper that all three modes call with different
sizing knobs. ``sample_template`` is the canonical reference for this
contract.
```

After editing, verify both worldview smoke tests still pass.
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader &amp;&amp; source venv/bin/activate &amp;&amp; python -c "
import sys
sys.path.insert(0, 'research')
# Reload to pick up the edit
import importlib
import dsplot.figures
importlib.reload(dsplot.figures)
# Convention text must mention all three modes
src = open('research/dsplot/figures/__init__.py').read()
for token in ('render(', 'show(', 'embed(', '_build_figure'):
    assert token in src, f'missing {token!r} in convention docstring'
# Existing re-exports still work
from dsplot.figures import figure_1, figure_2_4_1
print('existing re-exports ok')
# Optional new re-export (if added)
try:
    from dsplot.figures import sample_template
    print('sample_template re-exported')
except ImportError:
    print('sample_template not re-exported (acceptable)')
print('Task 2 OK')
" &amp;&amp; python -c "from research.dsplot.figures.sample_template import render, show, embed; print('B ok')"</automated>
  </verify>
  <done>
- `research/dsplot/figures/__init__.py` docstring contains the three-mode contract paragraph naming `render`, `show`, `embed`, `_build_figure`.
- Existing `figure_1` / `figure_2_4_1` re-exports untouched.
- (Optional) `sample_template` added to re-exports + `__all__`.
- Worldview A and Worldview B imports both work.
- NO git commits made.
  </done>
</task>

<task type="auto">
  <name>Task 3: Render v1.png + run both worldview smoke tests + scaffold dsp.ipynb cell (or SUMMARY.md fallback snippet)</name>
  <files>assets/images/dsp/figures/sample_template/v1.png, src/subshader/dsp/dsp.ipynb (or SUMMARY.md if ipynb edit risky)</files>
  <action>
NO git commits.

**Step 1 — Render the PNG:**
```bash
cd /home/eddie-water/dev/python/sub-shader && source venv/bin/activate
python -c "
import sys; sys.path.insert(0, 'research')
from dsplot.figures import sample_template
import matplotlib; matplotlib.use('Agg')
out = sample_template.render()
print('rendered to:', out)
"
```
Confirm the PNG lands at exactly `/home/eddie-water/dev/python/sub-shader/assets/images/dsp/figures/sample_template/v1.png`. If the path differs, fix the `render()` default args.

**Step 2 — Run both worldview smoke tests in isolation:**
```bash
# Worldview A — notebook-style sys.path shim
python -c "import sys; sys.path.insert(0, 'research'); from dsplot.figures import sample_template; sample_template.render(); print('A ok')"

# Worldview B — repo-root absolute imports
python -c "from research.dsplot.figures.sample_template import render; print(render()); print('B ok')"
```
Both must print their ok marker. If worldview B fails, the most likely cause is a `from research.X import …` lurking in sample_template.py — grep and fix.

**Step 3 — dsp.ipynb integration:**

Attempt to add a cell to `src/subshader/dsp/dsp.ipynb` calling `sample_template.show()`. Use a JSON-safe Python helper rather than free-form editing the file:

```python
import json
from pathlib import Path

NB_PATH = Path("src/subshader/dsp/dsp.ipynb")
nb = json.loads(NB_PATH.read_text())

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# dsplot — Sample Template (kitchen-sink showcase)\n",
        "from dsplot.figures import sample_template\n",
        "sample_template.show()\n",
    ],
}

# Skip if a cell already calls sample_template.show — idempotency safeguard.
already_present = any(
    isinstance(c.get("source"), list)
    and any("sample_template.show" in line for line in c["source"])
    for c in nb.get("cells", [])
)
if not already_present:
    nb.setdefault("cells", []).append(new_cell)
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print("appended cell")
else:
    print("cell already present — no-op")
```

**Risk gate — if any of the following are true, ABORT the ipynb edit and surface the snippet in SUMMARY.md for the user to paste manually instead:**
- `json.loads` on the existing ipynb raises (file may have unsaved widget state or is malformed).
- The notebook contains custom widget metadata at the top-level (`metadata.widgets`) that round-tripping through `json.dumps(indent=1)` would reformat in a way that breaks the user's render state.
- The notebook is currently open in JupyterLab (you can't detect this directly, but if a `.dsp.ipynb~` or lock file exists alongside it, that's a hint).

If aborted, include this exact snippet in the SUMMARY.md "ipynb integration" section:

> Paste into a new cell in `src/subshader/dsp/dsp.ipynb`:
>
> ```python
> from dsplot.figures import sample_template
> sample_template.show()
> ```

**Step 4 — Verify the rendered PNG is non-trivial:**
```bash
python -c "
import os
p = 'assets/images/dsp/figures/sample_template/v1.png'
assert os.path.exists(p), 'PNG missing'
sz = os.path.getsize(p)
assert sz > 50_000, f'PNG suspiciously small ({sz} bytes) — render may have failed silently'
print(f'PNG ok: {sz:,} bytes at {os.path.abspath(p)}')
"
```
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader &amp;&amp; source venv/bin/activate &amp;&amp; python -c "import sys; sys.path.insert(0, 'research'); from dsplot.figures import sample_template; sample_template.render(); print('A ok')" &amp;&amp; python -c "from research.dsplot.figures.sample_template import render; print(render()); print('B ok')" &amp;&amp; python -c "
import os
p = 'assets/images/dsp/figures/sample_template/v1.png'
assert os.path.exists(p), 'PNG missing'
sz = os.path.getsize(p)
assert sz > 50_000, f'PNG suspiciously small ({sz} bytes)'
print(f'PNG ok: {sz:,} bytes')
print('absolute path:', os.path.abspath(p))
"</automated>
  </verify>
  <done>
- `assets/images/dsp/figures/sample_template/v1.png` exists and is > 50 KB.
- Worldview A and Worldview B smoke tests both print their ok marker.
- Either: (a) `src/subshader/dsp/dsp.ipynb` has a new cell calling `sample_template.show()` and the notebook is valid JSON, OR (b) SUMMARY.md surfaces the paste snippet under "ipynb integration".
- The absolute PNG path is captured for SUMMARY.md.
- NO git commits made.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>
- NEW module `research/dsplot/figures/sample_template.py` — kitchen-sink dsplot showcase exercising every panel kind, every plottable, the explicit SuptitlePanel composition path, the Q2 cross-type overlay contracts (Line-on-Heatmap in bin-space + TimeSeriesPanel twin-axis with Line), and variable cell spans (1×1, 1×2, 1×3, 2×1). Five rows × 6-unit width, `show_cell_borders=True`.
- The `render() / show() / embed(target)` three-mode invocation contract is implemented in sample_template AND documented as the canonical convention for every future figure module in the `research/dsplot/figures/__init__.py` module docstring.
- Rendered PNG at `/home/eddie-water/dev/python/sub-shader/assets/images/dsp/figures/sample_template/v1.png`.
- `src/subshader/dsp/dsp.ipynb` either has a new `sample_template.show()` cell, or SUMMARY.md surfaces the paste snippet (whichever was safe).
- ZERO git commits — all changes left as working-tree edits per [[feedback-no-auto-commits]].
  </what-built>
  <how-to-verify>
1. **Open the PNG** (path will be printed in SUMMARY.md / Task 3 output, expected: `/home/eddie-water/dev/python/sub-shader/assets/images/dsp/figures/sample_template/v1.png`). Eyeball it for:
   - The SuptitlePanel row reads "Sample Template — dsplot Showcase" at the top, single line, centered, matching the v52 lego_demo title style (32pt, gray, bold).
   - Five rows are visible with `show_cell_borders=True` cell outlines so the layout structure is obvious.
   - All four panel kinds visibly populated: StaticPanel with 2D vector + components, StaticPanel3D with the 3D vector + dashed projections, DynamicPanel with the first frame of a sine, InteractivePanel with the static heatmap frame + slider bar at the bottom.
   - The TimeSeriesPanel in row 1 col 2 shows a twin y-axis on the right (Hz scale) with an orange instantaneous frequency Line overlaid on the gray TimeSeries.
   - The HeatmapPanel in row 1 col 3 shows a 2D field with a Line overlay drawn in bin-space (the highlight-colored track curving across the heatmap).
   - The CompositePanel in row 2 col 2 contains a Stem plot with a Dropline + Spotlight + Annotation all pointing at the peak sample.
   - Margins, gutters, fonts all read as consistent with the v52 lego_demo language — no visible style regressions vs `assets/images/dsp/lego_demo_v13.png` for shared elements (panel borders, tick lengths, subtitle weight).

2. **Spot-check the layout for anomalies:**
   - No empty cells.
   - No clipped tick labels at the figure edges.
   - The DynamicPanel and InteractivePanel cells should NOT look broken — they render their first frame statically.
   - The InteractivePanel's prev/next button bar will be visible at the cell bottom (this is expected, the `_base_bottom_pad=0.18` reservation).

3. **If you have time, render `v2.png` to confirm iteration works:**
   ```bash
   cd /home/eddie-water/dev/python/sub-shader && source venv/bin/activate
   python -c "import sys; sys.path.insert(0, 'research'); from dsplot.figures import sample_template; sample_template.render(output_filename='v2.png')"
   ```
   This is optional — the v1.png is the canonical deliverable.

4. **(Optional) Open dsp.ipynb and run the new cell** to confirm `sample_template.show()` displays inline correctly in the notebook. If the ipynb edit was skipped (per the Task 3 risk gate), the SUMMARY.md "ipynb integration" section contains the paste snippet — paste it into a new cell and run it.

5. **Confirm zero executor commits:**
   ```bash
   git log --oneline -5    # head should be unchanged from session start
   git status              # shows the new + modified files as working-tree edits
   ```

6. **If everything reads correctly:** type "approved" — the user will then decide separately when/whether to commit the working-tree changes.
   **If anything regresses:** describe the issue (which row/cell, what looks wrong) so the next iteration can address it via a v2 render.
  </how-to-verify>
  <resume-signal>Type "approved" or describe issues (which row/cell, what's wrong, what should change)</resume-signal>
</task>

</tasks>

<verification>
Phase-level: every must-haves truth confirmed by either the automated verify in Task 1/2/3 or by the human-verify checkpoint in Task 4.

Cross-task invariants checked:
- Width-equality across all rows = 6 units (Figure.compose validates internally; if mismatched the build fails in Task 1's verify).
- Both worldview smoke tests pass (verified in Task 3's automated block).
- No `from research.X import` inside sample_template.py (verified by Worldview A passing — that path breaks on `from research.X`).
- No hardcoded style literals (developer-discipline checked at authoring time per the conventions block).
- Zero executor commits (verified by reading `git log` at checkpoint time).
</verification>

<success_criteria>
- `research/dsplot/figures/sample_template.py` exists with the kitchen-sink layout described above.
- `research/dsplot/figures/__init__.py` has the three-mode contract paragraph documenting render/show/embed as the canonical convention.
- `assets/images/dsp/figures/sample_template/v1.png` exists, > 50 KB, and visually represents every panel + every plottable + the SuptitlePanel composition path + the Q2 cross-type overlays.
- Worldview A smoke test prints "A ok".
- Worldview B smoke test prints "B ok".
- Either dsp.ipynb has a `sample_template.show()` cell OR SUMMARY.md surfaces the paste snippet.
- Human checkpoint receives "approved" (or feedback for v2).
- ZERO `git commit` invocations by the executor — `git log --oneline -5` head is identical to the value at session start, all changes appear in `git status` as working-tree edits.
- SUMMARY.md surfaces the absolute PNG path + proposed commit message + file list for the user to commit at their own discretion.
</success_criteria>

<output>
Create `.planning/quick/260521-tpz-master-sample-template-figure-exercising/260521-tpz-SUMMARY.md` when done. The SUMMARY.md must include:
- The absolute path to `v1.png`.
- Smoke-test output (A ok / B ok lines).
- Coverage audit confirming every panel kind + every plottable is exercised at least once.
- The ipynb integration outcome (cell added OR paste snippet for the user).
- A "Files NOT committed" section listing all working-tree modifications + a suggested single commit message for the user to use at their discretion.
- The 4×4 span skip note (deferred from v1 per scope_brief allowance).
- Any deviations from the plan (e.g. if Spotlight `mode="band"` wasn't supported and the row-3-col-3 panel had to switch to `mode="scatter"`).
- A "Deferred / open questions" section noting that `embed(target: Figure)` and `embed(target: Axes)` are NotImplementedError stubs in v1, reserved for future simpler single-panel figure modules.
</output>
