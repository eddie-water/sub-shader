---
phase: quick-260518-tts
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - research/dsplot/figures/style_skeleton.py
  - assets/images/dsp/style_skeleton.png
autonomous: true
requirements:
  - QUICK-260518-TTS-01
must_haves:
  truths:
    - "research/dsplot/figures/style_skeleton.py imports cleanly under matplotlib Agg backend"
    - "Calling build_figure() returns a configured but un-rendered dsplot.Figure with a 2x3 layout and width_ratios=[1, 2, 1]"
    - "Calling fig.render() then fig.savefig(...) produces assets/images/dsp/style_skeleton.png"
    - "All 7 plottables (TimeSeries, Heatmap, Spotlight, Annotation, Dropline, Vector, VectorComponents) appear at least once across the 6 panels"
    - "All 4 panel types (StaticPanel, StaticPanel3D, DynamicPanel, InteractivePanel) appear, one per role; StaticPanel is reused across the two 2D static cells"
    - "The module contains zero literal style values — every color/font_size/linewidth/alpha/dpi/margin/hspace/wspace reads from dsplot.style.*"
    - "Working tree at commit time contains exactly two changed files: the new .py and the new .png; PLAN.md, SUMMARY.md, STATE.md are NOT in the commit"
  artifacts:
    - path: "research/dsplot/figures/style_skeleton.py"
      provides: "Canonical kitchen-sink reference figure module for dsplot — every panel type and plottable exercised, 100% style-driven"
      exports: ["build_figure", "show"]
    - path: "assets/images/dsp/style_skeleton.png"
      provides: "Rendered PNG of the skeleton figure for visual review"
  key_links:
    - from: "research/dsplot/figures/style_skeleton.py"
      to: "dsplot.style"
      via: "every style attribute reads through style.*"
      pattern: "style\\."
    - from: "research/dsplot/figures/style_skeleton.py"
      to: "dsplot.Figure(width_ratios=...)"
      via: "Figure constructor wired with 2x3 + width_ratios=[1, 2, 1]"
      pattern: "width_ratios"
---

<objective>
Build ONE new file: `research/dsplot/figures/style_skeleton.py` — a canonical kitchen-sink reference figure for dsplot. Every panel type, every plottable, every style knob exercised; zero literal styling values. Future dsplot figure authors open this file to see what a compliant figure looks like.

Purpose: Establishes a tangible "style discipline" exemplar — locks in the D-05 contract (style flows from `dsplot.style.*`, never from literals) into a single inspectable module that mirrors the idiom from `components_recombine.py` and the terseness of `style_override_demo.py`.

Output:
  - `research/dsplot/figures/style_skeleton.py` (new module)
  - `assets/images/dsp/style_skeleton.png` (rendered verification artifact)
</objective>

<execution_context>
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/workflows/execute-plan.md
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@research/dsplot/style.py
@research/dsplot/__init__.py
@research/dsplot/figure.py
@research/dsplot/figures/components_recombine.py
@research/dsplot/figures/style_override_demo.py
@research/dsplot/panels/static_panel.py
@research/dsplot/panels/dynamic_panel.py
@research/dsplot/panels/static_panel_3d.py
@research/dsplot/panels/interactive_panel.py
@research/dsplot/plottables/time_series.py
@research/dsplot/plottables/heatmap.py
@research/dsplot/plottables/spotlight.py
@research/dsplot/plottables/annotation.py
@research/dsplot/plottables/dropline.py
@research/dsplot/plottables/vector.py
@research/dsplot/plottables/vector_components.py

<interfaces>
<!-- Key contracts the executor needs. Extracted from the source files above. -->
<!-- The executor uses these directly — no codebase exploration needed. -->

From dsplot (`research/dsplot/__init__.py`):
```python
from dsplot import (
    style,            # the inheritable style template
    Figure,           # composes panels into one matplotlib Figure
    StaticPanel,      # single-state 2D panel
    StaticPanel3D,    # single-state 3D panel
    DynamicPanel,     # multi-frame FuncAnimation panel
    InteractivePanel, # widget-driven (slider + prev/next) panel
    Vector,           # polymorphic 2D / 3D arrow
    VectorComponents, # 2D-only x/y decomposition with optional droplines
    TimeSeries,       # 1D signal vs time via fill_between
    Heatmap,          # 2D imshow
    Spotlight,        # rectangle / scatter / glow highlight
    Annotation,       # text (+ optional arrow)
    Dropline,         # dashed perpendicular indicator
)
```

`Figure` constructor (`research/dsplot/figure.py`) — width_ratios IS supported:
```python
Figure(
    *,
    n_rows: int = 1, n_cols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    suptitle_fontsize: Optional[float] = None,
    hspace: Optional[float] = None,    # None -> style.DEFAULT_HSPACE
    wspace: Optional[float] = None,    # None -> style.DEFAULT_WSPACE
    width_ratios: Optional[List[float]] = None,    # <-- THIS ONE
    height_ratios: Optional[List[float]] = None,
    dpi: Optional[int] = None,         # None -> style.DEFAULT_DPI
    ...
)
fig.add_panel(panel, row=, col=, projection=None|"3d")
fig.render()
fig.savefig(path)
fig.close()
```

`StaticPanel`:
```python
StaticPanel(
    *,
    title: str | None = None,
    subtitle: str | None = None,
    lim: float | tuple | None = None,
    axis_style: str = "line",   # "arrow" for cartesian-arrow axes (vector cells)
    axis_labels: bool = False,
    show_border: bool = True,
)
panel.add(plottable)
```

`StaticPanel3D`:
```python
StaticPanel3D(
    *,
    lim_3d: float = 1.0,
    view_init: tuple[float, float] = (30.0, -60.0),
    title: str | None = None,
    subtitle: str | None = None,
    show_spines: bool = True,
)
# 2-tuple Vector on a StaticPanel3D auto-extends to (x, y, 0) (D-06).
# 3-tuple Vector on a 2D Axes is a TypeError.
```

`DynamicPanel` (pre-computed frames path):
```python
DynamicPanel(
    *,
    frames: list[list[Plottable]],
    interval_ms: int = 250,
    repeat: bool = True,
    base_plottables: list | None = None,
    # plus the StaticPanel-style chrome kwargs:
    title, subtitle, lim, axis_style, axis_labels, show_border,
)
# When wrapped by Figure, the figure installs a master clock and ticks each
# DynamicPanel via tick(); this happens automatically — no extra wiring.
```

`InteractivePanel`:
```python
InteractivePanel(
    *,
    frames: list[list[Plottable]],     # at least 1
    slider: bool = False,              # set True for slider sweep
    checkbox: tuple[str, callable] | None = None,
    # plus the StaticPanel-style chrome kwargs (title, subtitle, lim, ...).
)
```

Plottable signatures (only the kwargs this plan uses):
```python
Vector(vec, *, origin=None, color=None, linewidth=None, alpha=1.0,
       linestyle="-", label=None, label_offset=None, show_tip=True, zorder=2)
VectorComponents(vec, *, first_axis="x", show_droplines=True,
                 component_color=None, dropline_color=None,
                 label_x=None, label_y=None, alpha=0.95, zorder=2)
TimeSeries(signal, sample_rate, *, color=None, alpha=0.75,
           ylim_padding=1.1, zorder=2)
Heatmap(data, *, duration_s=None, freqs=None, log_freq=False,
        cmap=None, vmin=0.0, vmax=None, vmax_percentile=None,
        origin="lower", aspect="auto", zorder=1)
Spotlight(*, mode="rectangle"|"scatter"|"glow",
          x_range=None, y_range=None, xy=None, radius=0.05,
          color=None, alpha=0.35, linewidth=1.5, linestyle="--", zorder=6)
Annotation(text, xy, *, arrow_to=None, color=None, fontsize=None,
           fontweight="normal", ha="center", va="center",
           transform="data"|"axes", alpha=1.0, zorder=5)
Dropline(start, end, *, color=None, alpha=None, linewidth=None,
         linestyle=None, zorder=1)
```

Available style constants (the only legal source of styling — `research/dsplot/style.py`):
```
# palette (role-named)
PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, NEUTRAL_COLOR,
HIGHLIGHT_COLOR, BG_COLOR, SPINE_COLOR, TICK_LABEL_COLOR, DROPLINE_COLOR
# linework / arrows
DEFAULT_VECTOR_LINEWIDTH, DEFAULT_VECTOR_BOLD_LINEWIDTH,
DEFAULT_SPINE_LINEWIDTH, DEFAULT_DROPLINE_LINEWIDTH,
DEFAULT_DROPLINE_ALPHA, DEFAULT_DROPLINE_LINESTYLE,
DEFAULT_ARROW_HEAD_LENGTH, DEFAULT_ARROW_HEAD_WIDTH, DEFAULT_ARROW_MUTATION
# typography
DEFAULT_TITLE_FONT_SIZE, DEFAULT_SUBTITLE_FONT_SIZE,
DEFAULT_LABEL_FONT_SIZE, DEFAULT_SUBTITLE_Y, DEFAULT_TITLE_Y,
DEFAULT_SUBTITLE_BOTTOM_PAD, DEFAULT_TICK_LABEL_SIZE,
DEFAULT_AXIS_LABEL_SIZE, DEFAULT_SUPTITLE_FONT_SIZE, DEFAULT_ROW_LABEL_SIZE
# sizing / layout
DEFAULT_DPI, DEFAULT_PANEL_SIZE_INCHES, DEFAULT_PANEL_MARGIN,
DEFAULT_HSPACE, DEFAULT_WSPACE, DEFAULT_LABEL_RATIO,
# vector axes
DEFAULT_VECTOR_LABEL_OFFSET, DEFAULT_VECTOR_LIM,
DEFAULT_AXIS_GRID_COLOR, DEFAULT_AXIS_GRID_ALPHA,
DEFAULT_AXIS_GRID_LINEWIDTH, DEFAULT_AXIS_ARROW_INSET,
DEFAULT_AXIS_LABEL_OFFSET,
# heatmap
DEFAULT_HEATMAP_CMAP, DEFAULT_HEATMAP_VMAX_PERCENTILE
```

Pattern to mirror — show() invocation idiom from `components_recombine.py` lines 581-593:
```python
def show() -> Figure:
    import matplotlib.pyplot as plt
    fig = build_figure()
    fig.render()
    plt.show()
    return fig
```

Pattern to mirror — terseness of `style_override_demo.py`:
- Top-of-file docstring ~3-5 lines, no library-mechanics explanation.
- No `__main__` block (the demo's `__main__` exists because it's a CLI; this skeleton is NOT a CLI).
- Helper functions named `_panel_<role>()`; no inline-mega-function.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write research/dsplot/figures/style_skeleton.py — canonical 2x3 kitchen-sink figure</name>
  <files>research/dsplot/figures/style_skeleton.py</files>
  <action>
    Create `research/dsplot/figures/style_skeleton.py`. The module is a single source file that builds a 2-row x 3-column dsplot Figure with `width_ratios=[1, 2, 1]` so the middle column (wide signals: time-series, animated time-series) is 2x the width of the flanking columns (vectors, heatmaps, 3D).

    File layout (top-to-bottom, mirroring the style of `components_recombine.py`):

    1. Module docstring (3 lines max — title line + 1-2 lines on role). DO NOT explain matplotlib semantics or library mechanics.
    2. `from __future__ import annotations` and the dsplot imports — mirror the import block in `components_recombine.py` (single `from dsplot import (...)` grouped import).
    3. NumPy import (`import numpy as np`) — needed for dummy sinusoid + 2D gaussian + parameter sweep field.
    4. `os` import — needed to compute the absolute output PNG path inside `render()` / verification.
    5. A small constants block for the dummy CONTENT values only (these are data, not style — they are the only literal numbers allowed):
       - `A2 = (2.0, 3.0)` — the 2D dummy vector
       - `A3 = (2.0, 3.0, 1.0)` — the 3D dummy vector
       - `LIM_2D = 4.0` — symmetric 2D axes limit (chosen to fit A2 with headroom; document inline why)
       - `LIM_3D = 4.0` — 3D axes limit
       - `SAMPLE_RATE = 1000.0` (Hz, content)
       - `DURATION = 1.0` (s, content)
       - `SIGNAL_FREQ = 2.0` (Hz — ~2 cycles over DURATION)
       - `GRID_N = 32` — heatmap grid size
       - `BUILDUP_FRAMES = 5` — DynamicPanel frame count
       - `SLIDER_FRAMES = 5` — InteractivePanel sweep length (sigma sweep)
    6. Six panel-builder helpers, one per cell. Each returns a configured (but not rendered) Panel. Order:

       - `_panel_static_vector_arrow_axes()` -> StaticPanel
         - `axis_style="arrow"`, `axis_labels=True`, `show_border=False`, `lim=LIM_2D`, title="Vectors + Components".
         - Adds: `Dropline(A2, (A2[0], 0))`, `Dropline(A2, (0, A2[1]))`, `VectorComponents(A2, first_axis="x", show_droplines=False, component_color=style.NEUTRAL_COLOR)`, `Vector(A2, color=style.PRIMARY_COLOR, label="a", linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)`, `Annotation("a", xy=(A2[0]+0.30, A2[1]/2), color=style.PRIMARY_COLOR, fontsize=style.DEFAULT_LABEL_FONT_SIZE, fontweight="bold", ha="left", va="center")`.

       - `_panel_static_timeseries()` -> StaticPanel  (THIS IS THE WIDE [0,1] CELL)
         - `axis_style="line"`, `show_border=True`, title="Time Series + Spotlight".
         - Build a dummy sinusoid: `t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE`, `signal = np.sin(2 * np.pi * SIGNAL_FREQ * t)`.
         - Find the global maximum sample's time (`t_peak = float(t[int(np.argmax(signal))])`, `y_peak = float(np.max(signal))`).
         - Adds: `TimeSeries(signal, SAMPLE_RATE, color=style.PRIMARY_COLOR)`, `Dropline(start=(t_peak, y_peak), end=(t_peak, 0.0))`, `Spotlight(mode="scatter", xy=(t_peak, y_peak))`, `Annotation("peak", xy=(t_peak, y_peak), arrow_to=(t_peak, y_peak), color=style.HIGHLIGHT_COLOR, fontsize=style.DEFAULT_LABEL_FONT_SIZE, fontweight="bold")`. (`arrow_to` differs from `xy` is what triggers the arrow callout; an offset xy with arrow_to=(peak) is the right shape — see Annotation.draw().)

       - `_panel_static_heatmap()` -> StaticPanel
         - Build a 2D gaussian centered at the grid midpoint: use np.meshgrid on `np.linspace(-1, 1, GRID_N)` and `data = np.exp(-(X**2 + Y**2) / (2 * 0.3**2))`.
         - `show_border=True`, title="Heatmap + Annotation".
         - Adds: `Heatmap(data)` and `Annotation("centroid", xy=(GRID_N/2, GRID_N/2), color=style.HIGHLIGHT_COLOR, fontsize=style.DEFAULT_LABEL_FONT_SIZE, fontweight="bold")`. (Heatmap default extent is [0, w, 0, h] so the annotation's data coords match the bin grid.)

       - `_panel_static_3d_vector()` -> StaticPanel3D
         - `lim_3d=LIM_3D`, title="3D Vector".
         - Adds: `Vector(A3, color=style.PRIMARY_COLOR, label="a", linewidth=style.DEFAULT_VECTOR_BOLD_LINEWIDTH)`, plus three droplines projecting the tip onto each axis. Use polymorphic Vector with 3-tuples for the droplines: `Vector((0, 0, -A3[2]), origin=A3, color=style.DROPLINE_COLOR, linestyle="--", show_tip=False, alpha=style.DEFAULT_DROPLINE_ALPHA, linewidth=style.DEFAULT_DROPLINE_LINEWIDTH)` for the z-drop, and the analogous projections to land at (A3[0], A3[1], 0)->(A3[0], 0, 0) and (A3[0], A3[1], 0)->(0, A3[1], 0). (The Dropline Plottable is 2D-only; in 3D, dashed Vector instances with `show_tip=False` are the established idiom — see StaticPanel3D._draw_spines for the precedent.)

       - `_panel_dynamic_buildup()` -> DynamicPanel  (THIS IS THE WIDE [1,1] CELL)
         - 5-frame cumulative buildup of the same sinusoid: each frame extends the visible portion further. Frame k (0..4) shows the leading `int((k+1)/BUILDUP_FRAMES * len(signal))` samples padded with zeros to full length so the x-axis stays stable.
         - `axis_style="line"`, `show_border=True`, title="Animated Buildup", subtitle="cumulative sinusoid", `interval_ms=400`.
         - Each frame is `[TimeSeries(window_k, SAMPLE_RATE, color=style.PRIMARY_COLOR)]`.

       - `_panel_interactive_sigma_sweep()` -> InteractivePanel
         - 5-frame sigma sweep over the 2D gaussian: `sigmas = np.linspace(0.15, 0.45, SLIDER_FRAMES)`.
         - For each sigma, compute `data_k = np.exp(-(X**2 + Y**2) / (2 * sigma**2))` and put `[Heatmap(data_k)]` as the frame.
         - `slider=True`, `show_border=True`, title="Sigma Sweep", subtitle="interactive Heatmap".

    7. `build_figure()` -> dsplot.Figure
       - Create the Figure: `n_rows=2, n_cols=3, width_ratios=[1, 2, 1], suptitle="dsplot — Canonical Style Skeleton"`.
       - DO NOT pass `figsize`, `dpi`, `hspace`, or `wspace` — those resolve against `style.DEFAULT_*` by default (per D-05); passing them would be a style-discipline violation.
       - Add the six panels at their (row, col) positions. Use `projection="3d"` on the (1, 0) StaticPanel3D cell.
       - Return the configured but un-rendered Figure (do NOT call `fig.render()` here — `show()` does that).

    8. `show()` -> dsplot.Figure (mirror lines 581-593 of components_recombine.py exactly in shape):
       ```python
       def show() -> "Figure":
           """Build, render, and display the style skeleton figure."""
           import matplotlib.pyplot as plt
           fig = build_figure()
           fig.render()
           plt.show()
           return fig
       ```

    HARD RULES (style discipline — these are non-negotiable acceptance criteria, enforced by the verify step):
    - The ONLY literal numbers in the module are the dummy-content constants from item 5 above PLUS unavoidable math constants in dummy data construction (the `2 * np.pi` in the sinusoid, the `0.3` / `0.15` / `0.45` sigma values, `(k+1)/BUILDUP_FRAMES` fractions for the buildup, `LIM_2D`/`LIM_3D`-fitting Annotation offsets like `+0.30`). These are CONTENT/MATH, not STYLE.
    - Every color value, font size, linewidth, alpha (except plottable-default alphas explicitly None), DPI, margin, hspace, wspace MUST come from `dsplot.style.*`. Zero literal hex colors. Zero literal font sizes. Zero literal linewidths.
    - NO `__main__` block.
    - NO CLI argument parsing.
    - Module-level docstring is 3-5 lines max — no explanation of matplotlib semantics or library mechanics. Mirror the terseness of `style_override_demo.py`'s top-level docstring (lines 1-33 are fine to study but the canonical-skeleton docstring should be MUCH shorter; that demo's verbosity exists because it's pedagogical for D-05).
    - No `import` of any matplotlib internals beyond `matplotlib.pyplot as plt` inside `show()`.
    - No new entries in any `__init__.py`, no notebook touch, no manifest.

    DO NOT:
    - Touch `research/dsplot/figures/__init__.py` or `__main__.py`.
    - Touch `dsp.ipynb`.
    - Add docstrings longer than ~3 lines to any helper.
    - Add inline comments explaining matplotlib or library internals.
    - Add an `if __name__ == "__main__"` block.

    NOTE on dummy-vector limits: A2 = (2.0, 3.0) fits inside LIM_2D=4.0 with comfortable padding for the right-side Annotation offset. A3 = (2.0, 3.0, 1.0) fits inside LIM_3D=4.0 with the same logic.
  </action>
  <verify>
    <automated>
      cd /home/eddie-water/dev/python/sub-shader &amp;&amp; source venv/bin/activate &amp;&amp; \
      python - <<'PY'
import os, re, sys, ast

# --- 1. Headless import + render gate -------------------------------------
import matplotlib
matplotlib.use("Agg")

# Bootstrap the research/ path the same way style_override_demo.py does
THIS = os.path.abspath("research/dsplot/figures/style_skeleton.py")
RESEARCH_DIR = os.path.abspath("research")
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from dsplot.figures.style_skeleton import build_figure  # noqa: E402

fig = build_figure()
# Verify 2 rows x 3 cols + width_ratios were honored
gs = fig._gs
assert gs.nrows == 2 and gs.ncols == 3, f"want 2x3, got {gs.nrows}x{gs.ncols}"
wr = list(gs.get_width_ratios())
assert wr == [1, 2, 1] or wr == [1.0, 2.0, 1.0], f"width_ratios mismatch: {wr}"

# Render + save the PNG
fig.render()
out_path = "assets/images/dsp/style_skeleton.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path)
assert os.path.exists(out_path), f"PNG not written: {out_path}"
fig.close()

# --- 2. Plottable + panel coverage by static AST inspection ---------------
src = open(THIS).read()
tree = ast.parse(src)

# Names used (Call/Attribute) — easiest way to assert "this Plottable / Panel name appears"
called = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        # bare Name(...) or attr.Name(...)
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)

needed_plottables = {"Vector", "VectorComponents", "TimeSeries", "Heatmap",
                     "Spotlight", "Annotation", "Dropline"}
missing_plot = needed_plottables - called
assert not missing_plot, f"missing plottables: {missing_plot}"

needed_panels = {"StaticPanel", "StaticPanel3D", "DynamicPanel", "InteractivePanel"}
missing_pan = needed_panels - called
assert not missing_pan, f"missing panel types: {missing_pan}"

# --- 3. Style discipline gate (no literal hex / font-size / linewidth) ----
# Strip comments + strings before regex scan so docstrings can't carry literals
class _Stripper(ast.NodeTransformer):
    def visit_Constant(self, n):
        if isinstance(n.value, str):
            return ast.Constant(value="")
        return n
stripped = ast.unparse(_Stripper().visit(ast.parse(src)))

# 3a. No literal hex colors anywhere
hex_hits = re.findall(r'#[0-9a-fA-F]{3,8}\b', stripped)
assert not hex_hits, f"literal hex colors found: {hex_hits}"

# 3b. The plottable / panel / Figure kwargs that MUST be style-driven cannot
#     be passed as literal numbers. We scan every keyword= in every Call:
STYLE_KWARGS = {"fontsize", "linewidth", "dpi", "hspace", "wspace",
                "suptitle_fontsize", "head_length", "head_width",
                "mutation_scale"}
bad = []
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg in STYLE_KWARGS and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, (int, float)):
                bad.append((kw.arg, kw.value.value))
assert not bad, f"style kwargs passed as literals (must use style.*): {bad}"

# 3c. No literal color= hex/string anywhere; color= must be a Name or Attribute
bad_colors = []
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == "color" and isinstance(kw.value, ast.Constant):
                bad_colors.append(kw.value.value)
assert not bad_colors, f"literal color= values found (must use style.*): {bad_colors}"

# --- 4. No __main__ block / no CLI ----------------------------------------
assert '__main__' not in src, "__main__ block forbidden in style_skeleton.py"

# --- 5. Confirm exports -----------------------------------------------------
import importlib
mod = importlib.import_module("dsplot.figures.style_skeleton")
assert hasattr(mod, "build_figure"), "build_figure missing"
assert hasattr(mod, "show"),         "show missing"
assert callable(mod.build_figure) and callable(mod.show)

print("OK style_skeleton.py + style_skeleton.png")
PY
    </automated>
  </verify>
  <done>
    - `research/dsplot/figures/style_skeleton.py` exists, imports cleanly under Agg, exports `build_figure` and `show`.
    - `build_figure()` returns a 2x3 dsplot.Figure with `width_ratios=[1, 2, 1]`.
    - All 7 plottables (Vector, VectorComponents, TimeSeries, Heatmap, Spotlight, Annotation, Dropline) are constructed somewhere in the module.
    - All 4 panel types (StaticPanel, StaticPanel3D, DynamicPanel, InteractivePanel) are constructed.
    - Style-discipline AST gate passes: no literal hex colors; no literal numeric values passed as `fontsize`, `linewidth`, `dpi`, `hspace`, `wspace`, `suptitle_fontsize`, `head_length`, `head_width`, or `mutation_scale`; no literal string values passed as `color=`.
    - No `__main__` block.
    - `assets/images/dsp/style_skeleton.png` exists on disk.
  </done>
</task>

<task type="auto">
  <name>Task 2: Commit ONLY the new .py and the new .png (atomic)</name>
  <files>research/dsplot/figures/style_skeleton.py, assets/images/dsp/style_skeleton.png</files>
  <action>
    Make ONE atomic commit containing exactly two files:
      - `research/dsplot/figures/style_skeleton.py`
      - `assets/images/dsp/style_skeleton.png`

    Steps (use absolute paths; do not `cd`):
    1. Run `git status --porcelain=v1` and confirm the only modified/added entries you care about are the two target paths. PLAN.md, SUMMARY.md, STATE.md, and the planning quick directory MUST NOT appear in this commit — the orchestrator's Step 8 docs commit owns them. If a planning doc shows up as staged, unstage it with `git restore --staged <path>` before continuing.
    2. Stage exactly those two files by name (NEVER use `git add -A` or `git add .`):
       `git add research/dsplot/figures/style_skeleton.py assets/images/dsp/style_skeleton.png`
    3. Verify staging is exactly those two paths: `git diff --cached --name-only` must print exactly those two lines.
    4. Commit using a heredoc message:
       ```
       git commit -m "$(cat <<'EOF'
       feat(dsplot): add canonical style-skeleton reference figure

       2x3 kitchen-sink figure (width_ratios=[1, 2, 1]) exercising every dsplot
       panel type and every plottable, with style flowing entirely from
       dsplot.style.* (zero literal styling values).

       Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
       EOF
       )"
       ```
    5. Run `git status --porcelain=v1` again and confirm no leftover staged changes belonging to this task.

    DO NOT push. DO NOT merge. DO NOT touch any branch state besides the single commit. Stop after the commit.
  </action>
  <verify>
    <automated>
      cd /home/eddie-water/dev/python/sub-shader &amp;&amp; \
      git log -1 --name-only --pretty=format:'%s' | tee /tmp/style_skeleton_commit.txt &amp;&amp; \
      python - <<'PY'
txt = open('/tmp/style_skeleton_commit.txt').read().splitlines()
subject = txt[0]
files = sorted(l for l in txt[1:] if l.strip())
expected = sorted([
    "research/dsplot/figures/style_skeleton.py",
    "assets/images/dsp/style_skeleton.png",
])
assert files == expected, f"commit files mismatch:\n got: {files}\n want: {expected}"
assert subject.startswith("feat(dsplot)"), f"bad subject: {subject!r}"
print("OK atomic commit:", subject)
PY
    </automated>
  </verify>
  <done>
    - HEAD commit subject starts with `feat(dsplot)`.
    - HEAD commit contains EXACTLY the two files: `research/dsplot/figures/style_skeleton.py` and `assets/images/dsp/style_skeleton.png` (no more, no less).
    - No planning artifacts (PLAN.md / SUMMARY.md / STATE.md / .planning/quick/*) are in the commit.
    - Working tree clean of these two paths (`git status` does not list them as modified).
  </done>
</task>

</tasks>

<verification>
After both tasks complete, the executor must STOP and yield control. The user reviews `assets/images/dsp/style_skeleton.png` visually before any further action (no merge, no push, no follow-up plan). The PNG is the human-eyeball gate; the AST/import gate in Task 1 is the machine gate.

Phase-level success checks (all automated checks already encoded in Task 1's verify):
- Module imports under Agg without raising
- `build_figure()` produces a 2x3 Figure with width_ratios=[1, 2, 1]
- All 7 plottables + all 4 panel types appear in source
- AST scan finds zero literal style values in disallowed kwargs
- PNG renders to assets/images/dsp/style_skeleton.png
- Single atomic commit contains exactly the two target files
</verification>

<success_criteria>
- `research/dsplot/figures/style_skeleton.py` exists and is the sole new source file
- `assets/images/dsp/style_skeleton.png` exists and was produced by `build_figure()` -> `fig.render()` -> `fig.savefig(...)`
- The module's AST has zero literal styling values in the disallowed positions
- HEAD is one new commit with subject `feat(dsplot): add canonical style-skeleton reference figure` containing exactly those two files
- No planning artifacts committed alongside the source/image (PLAN.md, SUMMARY.md, STATE.md belong to the orchestrator's Step 8 docs commit)
- Executor has stopped — no push, no merge, no follow-up edits — pending user PNG review
</success_criteria>

<output>
Create `.planning/quick/260518-tts-add-canonical-dsplot-style-skeleton-figu/260518-tts-SUMMARY.md` when done (the orchestrator's Step 8 docs commit will include it — do NOT commit it as part of the atomic feat commit in Task 2).
</output>
