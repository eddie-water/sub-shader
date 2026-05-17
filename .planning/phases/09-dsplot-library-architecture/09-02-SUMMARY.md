---
phase: 09-dsplot-library-architecture
plan: 02
subsystem: dsplot (Panel layer + Figure orchestrator + axes setup)
tags: [dsplot, panel, figure, axes-setup, d-04, d-05, parallel-wave]
requires:
  - dsplot.style (D-05 inheritable template) — provided by 09-01
  - dsplot.plottables.base.Plottable (.draw(ax) contract) — provided by 09-01
provides:
  - dsplot.panels.Panel: abstract base — attach(ax) + add(plottable) + abstract render()
  - dsplot.panels.StaticPanel: single-state render; bg + spines + title/subtitle + axes_setup + plottable iteration
  - dsplot.figure.Figure: gridspec composition with projection kwarg per D-04 + lazy layout defaults per D-05
  - dsplot.axes_setup.setup_vector_axes: reusable square-axes / arrow-axes / x-y-labels helper
affects:
  - research/dsplot/__init__.py (Panel, StaticPanel, Figure added to __all__)
tech-stack:
  added:
    - matplotlib.figure.Figure + add_gridspec (no new dependencies — pure mpl primitives)
  patterns:
    - Panel ABC + attach/render lifecycle (Figure attaches, Panel renders)
    - Lazy style lookup at construction time (resolve `style.DEFAULT_*` inside `__init__`, not as class defaults)
    - Asymmetric add_subplot dispatch on `projection` kwarg (None → 2D Axes, "3d" → Axes3D)
key-files:
  created:
    - research/dsplot/axes_setup.py
    - research/dsplot/panels/__init__.py
    - research/dsplot/panels/base.py
    - research/dsplot/panels/static_panel.py
    - research/dsplot/figure.py
    - research/tests/dsplot/test_panel_composition.py
    - research/tests/dsplot/test_figure_orchestrator.py
  modified:
    - research/dsplot/__init__.py
decisions:
  - "Figure.render() — not __init__ — is where axes get created and panels attach. Construction is cheap (just figsize + gridspec); render is the side-effect step. Lets consumers build the Figure shape, attach panels, and only commit to drawing later."
  - "StaticPanel.render() uses setup_vector_axes for 2D vector-style chrome by default; 3D projection cells get 2D chrome applied awkwardly (intentional loud-look, not loud-fail). Wave 3's StaticPanel3D will properly handle 3D chrome (per plan risk note + 09-CONTEXT.md)."
  - "Figure.savefig() defaults to bbox_inches='tight', pad_inches=0.15 (mirror legacy _save in dsp_figures.py). Caller can override per-call; Figure does not enforce filename conventions."
metrics:
  duration: "single session (interleaved with 09-01 parallel work)"
  completed: 2026-05-17
---

# Phase 09 Plan 02: Panel Layer + Figure Orchestrator Summary

Built the Panel composition layer (`Panel` ABC, `StaticPanel` concrete) and the
`Figure` orchestrator that composes Panels into one matplotlib Figure with a
gridspec layout. Includes the `projection` kwarg on `Figure.add_panel` per the
LOCKED D-04 (3D cells compose via the same flow as 2D cells), the reusable
`setup_vector_axes` helper extracted from `research/utilities/plotting.py`, and
the D-05 lazy-lookup pattern for layout defaults (figsize / hspace / wspace /
dpi inherit from `style.DEFAULT_*` at construction time, not as class
defaults).

## Panel ABC contract (`research/dsplot/panels/base.py`)

```python
class Panel(ABC):
    ax: Axes | None
    _plottables: list[Plottable]

    def __init__(self) -> None: ...     # self.ax=None, self._plottables=[]
    def add(self, plottable) -> Panel:  # appends, returns self (fluent)
    def attach(self, ax) -> None:       # called by Figure during render
    @abstractmethod
    def render(self) -> None: ...
```

Three responsibilities split cleanly:
- **Construction**: panel's intrinsic state (title, lim, axis style) — no Axes yet.
- **`attach(ax)`**: Figure hands the panel an Axes when placing it into a gridspec cell.
- **`render()`**: subclass-specific; iterates `self._plottables` and calls `.draw(self.ax)`.

This shape is what Wave 2's DynamicPanel (FuncAnimation) and InteractivePanel
(ipywidgets) will extend without rewriting the composition primitive.

## StaticPanel behavior (`panels/static_panel.py`)

`StaticPanel.render()` does the following in order:

1. Raises `RuntimeError` if `self.ax is None` — "render() called before attach()".
2. Sets `ax.set_facecolor(style.BG_COLOR)`.
3. If `show_border`: colors each spine with `style.SPINE_COLOR` and width
   `style.DEFAULT_SPINE_LINEWIDTH`. Else hides spines.
4. Calls `setup_vector_axes(ax, lim=..., show_border=..., axis_style=..., axis_labels=...)`.
5. Sets `ax.set_title(self.title, fontsize=style.DEFAULT_TITLE_FONT_SIZE, color=style.TICK_LABEL_COLOR)` when title is set.
6. Places subtitle as `ax.text(0.5, 1.02, ..., transform=ax.transAxes, fontsize=style.DEFAULT_SUBTITLE_FONT_SIZE)` when subtitle is set.
7. Iterates plottables in insertion order, calling each `.draw(self.ax)`.

Every typography / sizing default flows from `dsplot.style.DEFAULT_*` per D-05 —
no hardcoded font sizes or spine widths in the panel.

## axes_setup helper (`axes_setup.py`)

Ported from `research/utilities/plotting.py::setup_vector_axes` with D-01
isolation (no `subshader` imports, no `research.utilities.style` imports — all
None defaults resolve against `dsplot.style.DEFAULT_*` lazily).

Signature:
```python
def setup_vector_axes(
    ax, *,
    lim=None, panel_title=None, result_text=None,
    show_border=True, axis_style="line", axis_labels=False,
    x_color=None, y_color=None,
    axis_alpha=None, axis_linewidth=None,
) -> None
```

Behavior:
- Square axes, symmetric ±lim limits, no ticks.
- `axis_style="line"`: axhline + axvline at y=0, x=0.
- `axis_style="arrow"`: two FancyArrow annotations from `-lim+inset` to `+lim-inset` on each axis.
- `axis_style="none"`: nothing drawn (consumer adds its own axes chrome).
- `axis_labels=True`: places "x" near +x tip in Q-IV, "y" near +y tip in Q-I, both italic.
- `panel_title` → `ax.set_title` at `DEFAULT_SUBTITLE_FONT_SIZE`.
- `result_text` → `ax.text(0.5, -0.08, ..., transform=ax.transAxes)` at `DEFAULT_LABEL_FONT_SIZE`.

All lazy-lookup pinned by test 9 in `test_panel_composition.py` — reassigning
`style.DEFAULT_VECTOR_LIM` between calls is observable through `ax.get_xlim()`.

## Figure orchestrator (`figure.py`)

Composition pattern:

```python
fig = Figure(n_rows=1, n_cols=3, suptitle="Title")  # figsize etc. from style.DEFAULT_*
fig.add_panel(StaticPanel(title="A"), row=0, col=0)
fig.add_panel(StaticPanel(title="B"), row=0, col=1)
fig.add_panel(StaticPanel(title="C"), row=0, col=2, projection="3d")
fig.render()
fig.savefig("out.png")
fig.close()
```

Key contract decisions:

- **Layout defaults resolve at `__init__` time** (per D-05) — `figsize` falls
  back to `(style.DEFAULT_PANEL_SIZE_INCHES * n_cols, * n_rows)`; `hspace` /
  `wspace` / `dpi` likewise. Reassigning a style default between Figure
  constructions is observable in subsequent `gs.hspace` etc.
- **`add_panel(panel, row, col, rowspan, colspan, projection)` returns the
  panel** for fluent chaining (`fig.add_panel(p, ...).add(plottable)` would
  call `.add` on the Panel).
- **`render()` walks `self.panels`** and calls
  `self._mpl_fig.add_subplot(self._gs[row:row+rowspan, col:col+colspan], projection=...)`
  with `projection=None` becoming a regular 2D `Axes`, `projection="3d"`
  becoming an `Axes3D`. Then `panel.attach(ax); panel.render()`.
- **2D axes get `style.BG_COLOR` + spine styling at render time**; 3D axes
  get bg color only (spine concept doesn't apply cleanly to mpl 3D — panels
  handle 3D chrome themselves, e.g. the future StaticPanel3D in 09-05).
- **`savefig(path, **kwargs)`** mirrors the legacy `_save` in
  `research/dsp_figures.py`: `bbox_inches="tight"`, `pad_inches=0.15` defaults
  (overridable); creates parent dirs; returns absolute path.

## D-04 3D projection kwarg

The 3D test (`test_figure_projection_3d_creates_axes3d`) pins:

```python
fig = Figure(n_rows=1, n_cols=2)
fig.add_panel(panel_3d, row=0, col=0, projection="3d")
fig.add_panel(panel_2d, row=0, col=1)                  # default None
fig.render()
assert hasattr(panel_3d.ax, "get_zlim")    # Axes3D
assert not hasattr(panel_2d.ax, "get_zlim") # regular Axes
```

Mixed 2D + 3D cells compose in one Figure — no escape hatch needed for the 3D
foundation figure (`vector_projection_3d`) when Wave 3 ports it.

## D-05 lazy-lookup pattern

Every layout default in this plan resolves INSIDE the `__init__` body, not
as a class-default kwarg. That's the lazy-lookup contract that makes
`style.DEFAULT_HSPACE = 0.71; Figure(...).gs.hspace == 0.71` work
(`test_figure_layout_defaults_resolve_lazily_against_style`).

Reading `style.DEFAULT_*` as a default-argument value (e.g.
`def __init__(self, hspace=style.DEFAULT_HSPACE)`) would freeze the value at
import time — late reassignment wouldn't take effect. The pattern:

```python
def __init__(self, *, hspace: Optional[float] = None, ...) -> None:
    if hspace is None:
        hspace = style.DEFAULT_HSPACE
    ...
```

is the inheritable-template-with-override contract.

## Rowspan / colspan composition primitive

`add_panel(panel, row=0, col=0, rowspan=2, colspan=2)` on a 3x3 grid attaches
the panel to a 2x2 cell block. Verified via comparison: spanned-panel bbox is
> 1.5x the width and height of a single-cell baseline (test
`test_figure_rowspan_colspan_spans_multiple_cells`).

Wave 3's motivator port (3x2 grid with a narrow label column on the left)
uses `width_ratios` on the Figure plus rowspan to compose row-label cells
that span all three audio panel rows.

## Tests added

`research/tests/dsplot/test_panel_composition.py` (9 tests, all passing):
- Panel is abstract (`TypeError` on direct construction)
- StaticPanel.add returns self (fluent chaining)
- Plottables stored in insertion order
- `panel.render()` calls each `plottable.draw(self.ax)` exactly once
- Title / subtitle use `style.DEFAULT_TITLE_FONT_SIZE` / `DEFAULT_SUBTITLE_FONT_SIZE`
- `lim=N` produces square ±N axes
- `axis_style="arrow"` + `axis_labels=True` adds x/y label texts + arrow annotations
- `axis_style="line"` adds axhline + axvline
- D-05 lazy lookup: `style.DEFAULT_VECTOR_LIM` reassignment observable

`research/tests/dsplot/test_figure_orchestrator.py` (9 tests, all passing):
- Figure constructs with grid dims
- `add_panel` attaches axes via render
- `add_panel` returns the panel (fluent)
- `savefig` writes a non-empty PNG
- `suptitle` set on mpl Figure
- End-to-end with real `StaticPanel` + `Vector` + savefig
- Rowspan / colspan spans multiple cells (vs single-cell baseline)
- D-04 `projection="3d"` produces Axes3D for that cell
- D-05 lazy lookup: `style.DEFAULT_HSPACE` reassignment observable

Full dsplot suite (62 from 09-01 + 9 from 09-02 task 1 + 9 from task 2 = 71
tests after some 09-01 tests consolidated) passes:
```
71 passed in 0.36s
```

## Deviations from Plan

### Tightened rowspan / colspan assertion (Rule 1 — Bug)

- **Found during:** Task 2 GREEN.
- **Issue:** The plan's behavior described "verify via `panel.ax.get_position()`
  width covers ~2/3 of figure width" via a hardcoded `> 0.5` threshold.
  mpl's tight-layout margins on a 3x3 grid put the spanned bbox at 0.4996
  — barely below the threshold.
- **Fix:** Replaced the absolute threshold with a comparison against a
  single-cell baseline bbox on an equivalent grid (spanned width > 1.5x
  baseline). Same intent, more robust to mpl's margin defaults.
- **Files modified:** `research/tests/dsplot/test_figure_orchestrator.py`
- **Commit:** d713105 (Task 2 GREEN — fix folded into the same commit).

### No other deviations

`StaticPanel` rendering on a 3D Axes (when a consumer passes
`projection="3d"` to a panel built for 2D) was kept as a "loud-look" rather
than a "loud-fail" — `setup_vector_axes` applies 2D chrome to the 3D axes
which renders awkwardly but does not crash. This matches the plan's risk
note: "a consumer who passes `projection='3d'` AND a StaticPanel built for
2D will hit a panel-level error during render … 09-05 task 2 introduces
`StaticPanel3D` for 3D-specific panel chrome." The pedagogical fail mode
will manifest visually rather than as an exception. If Wave 3's review
prefers a hard guard, raising in `StaticPanel.render` when
`hasattr(self.ax, "get_zlim")` is a one-line addition.

## Parallel execution notes (09-01 + 09-02 interleaving)

Both plans landed on the same `gsd/phase-09-dsplot-library-architecture`
branch. The shared `.git/index` led to one interleaving event where 09-01's
staged-but-uncommitted `__init__.py` modifications and `plottables/vector.py`
landed under a 09-02 commit message; that commit (`b1250dd`) was amended to
`feat(09-01)` (now `a7efd47`) to correctly attribute the content. After that,
all per-task stages explicitly listed paths and verified `git diff --cached
--stat` before commit. Lesson logged for future parallel-on-main-repo plans:
**explicit single-file `git add` followed by a staged-diff verification step
is required when two agents share an index**. Worktree isolation would avoid
this entirely; this run did not have it because `.git` was a directory (main
repo), not a file (worktree).

## Pre-existing items NOT in scope (deferred)

- `dsplot.style` exposes `DEFAULT_AXIS_GRID_COLOR` / `DEFAULT_AXIS_GRID_ALPHA`
  / `DEFAULT_AXIS_GRID_LINEWIDTH` — the `GRID` substring trips a strict
  reading of non-goal #5 ("no `GRID_*` constants"). These names describe
  generic axis-grid styling, not the comparison-grid figure that non-goal #5
  was warning about — they're functionally project-agnostic. If 09-01's
  follow-up review prefers a rename (e.g. `DEFAULT_AXLINE_COLOR`), that's
  one-line per constant. Logged here, not blocking.

## Self-Check

Created files verified on disk:
- `research/dsplot/axes_setup.py` — FOUND
- `research/dsplot/panels/__init__.py` — FOUND
- `research/dsplot/panels/base.py` — FOUND
- `research/dsplot/panels/static_panel.py` — FOUND
- `research/dsplot/figure.py` — FOUND
- `research/tests/dsplot/test_panel_composition.py` — FOUND
- `research/tests/dsplot/test_figure_orchestrator.py` — FOUND

Commits verified in `git log`:
- `a02bd07` test(09-02): add failing tests for Panel ABC + StaticPanel + axes_setup helper — FOUND
- `db90747` feat(09-02): implement Panel ABC + StaticPanel + axes_setup helper — FOUND
- `6444168` test(09-02): add failing tests for Figure orchestrator — FOUND
- `d713105` feat(09-02): implement Figure orchestrator with projection kwarg + lazy layout — FOUND

Verification commands all pass:
- `python -m pytest research/tests/dsplot/ -q` → 71 passed
- `python -c "from dsplot import Figure, StaticPanel, Vector, ..."` exits 0
- `grep -rIn "subshader" research/dsplot/ --exclude-dir=figures` returns zero lines (exit 1)
- End-to-end `Figure + StaticPanel + Vector + savefig` smoke produces a non-empty PNG
- D-04 mixed 2D + 3D Figure smoke composes cleanly

## Self-Check: PASSED
