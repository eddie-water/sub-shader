---
phase: quick-260521-svu
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - research/dsplot/panels/suptitle_panel.py
  - research/dsplot/panels/__init__.py
  - research/dsplot/__init__.py
  - research/dsplot/style.py
  - research/dsplot/figure.py
  - research/dsplot/plottables/heatmap.py
  - research/dsplot/plottables/line.py
  - research/dsplot/plottables/time_series.py
  - research/dsplot/panels/heatmap_panel.py
  - .planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md
autonomous: false
requirements:
  - SVU-G1-suptitle-panel
  - SVU-G2-cross-type-overlays
  - SVU-G3-pad-audit

must_haves:
  truths:
    - "Figure.compose(suptitle='...', rows=[...]) renders bit-identical to today (sugar preserved)"
    - "Figure.compose(rows=[[SuptitlePanel('...')], [PanelA, PanelB]], ...) renders an equivalent suptitle when passed explicitly"
    - "Width-equality guard raises ValueError when an explicit SuptitlePanel's row width doesn't match sibling rows"
    - "Heatmap accepts alpha kwarg and passes it through to ax.imshow()"
    - "Line and TimeSeries plottables accept zorder + alpha kwargs (already true) and Heatmap zorder default = 1, TimeSeries = 2, Line = 3"
    - "HeatmapPanel docstring documents the 'Line overlay shares the primary axis in bin-space; caller transforms' contract"
    - "figure_1.render_hero() renders without error and visually matches the prior hero output (no regression)"
    - "Both import worldviews continue to work (notebook + repo-root)"
  artifacts:
    - path: research/dsplot/panels/suptitle_panel.py
      provides: SuptitlePanel subclass
      contains: "class SuptitlePanel"
    - path: research/dsplot/style.py
      provides: SUPTITLE_* constants
      contains: "SUPTITLE_FONT_SIZE"
    - path: research/dsplot/panels/heatmap_panel.py
      provides: docstring describing Line-on-Heatmap bin-space contract
      contains: "primary axis"
  key_links:
    - from: research/dsplot/figure.py
      to: research/dsplot/panels/suptitle_panel.py
      via: "compose(suptitle=...) sugar path instantiates SuptitlePanel internally"
      pattern: "SuptitlePanel"
    - from: research/dsplot/plottables/heatmap.py
      to: matplotlib imshow
      via: "alpha kwarg pass-through"
      pattern: "alpha=self.alpha"
---

<objective>
Promote the figure suptitle into a real `SuptitlePanel` (preserving the
`Figure.compose(suptitle="...")` sugar bit-identically), verify cross-type
plottable overlays support `zorder`/`alpha` on Line/Heatmap/TimeSeries (filling
the one real gap: Heatmap's missing `alpha` kwarg), document the
Line-on-Heatmap bin-space overlay contract on HeatmapPanel, and run a
read-only pad invariant audit across panels + figure.

Purpose: Move the suptitle from a hardcoded `_mpl_fig.suptitle(...)` call into
the lego composition model so it participates in the panel-grid the same way
TextPanel/HeatmapPanel/etc. do — and lock the cross-type overlay pattern
figure_1 already depends on by giving every overlay plottable a uniform
zorder/alpha contract.

Output:
- New `SuptitlePanel` exported from `research.dsplot`
- New `SUPTITLE_*` style constants
- `Heatmap` plottable accepts `alpha=`
- HeatmapPanel docstring on overlay contract
- Light pad-invariant audit findings in SUMMARY.md
- Pre-existing visual output of `figure_1.render_hero()` unchanged
</objective>

<execution_context>
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/workflows/execute-plan.md
@/home/eddie-water/dev/python/sub-shader/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/eddie-water/dev/python/sub-shader/.planning/STATE.md
@/home/eddie-water/.claude/projects/-home-eddie-water-dev-python-sub-shader/memory/project_dsplot_framework.md
@/home/eddie-water/.claude/projects/-home-eddie-water-dev-python-sub-shader/memory/project_dsplot_import_worldviews.md
@/home/eddie-water/.claude/projects/-home-eddie-water-dev-python-sub-shader/memory/feedback_no_auto_commits.md
@/home/eddie-water/dev/python/sub-shader/research/dsplot/__init__.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/figure.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/style.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/panels/base.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/panels/text_panel.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/panels/composite_panel.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/panels/heatmap_panel.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/panels/time_series_panel.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/plottables/heatmap.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/plottables/line.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/plottables/time_series.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/figures/figure_1.py
@/home/eddie-water/dev/python/sub-shader/research/dsplot/figures/lego_demo.py

<interfaces>
<!-- Key contracts the executor needs. Extracted from current code. -->

Suptitle render call lives in `research/dsplot/figure.py` __init__ (~line 142):

```python
self._mpl_fig.suptitle(
    suptitle,
    color=style.TICK_LABEL_COLOR,
    fontsize=sup_fontsize_resolved,  # resolves to style.DEFAULT_SUPTITLE_FONT_SIZE (32)
    fontweight="bold",
    y=...,
    va="center",
)
```

So the current visual contract is: color=TICK_LABEL_COLOR (#888888), size=DEFAULT_SUPTITLE_FONT_SIZE (32), weight="bold". The new `SUPTITLE_*` constants must default to these exact values to preserve bit-identical output. Suggested constant names (mirroring TITLE_* pattern):
- `SUPTITLE_FONT_SIZE = DEFAULT_SUPTITLE_FONT_SIZE` (alias or new — keep DEFAULT_SUPTITLE_FONT_SIZE intact as the existing constant figure.py already reads)
- `SUPTITLE_COLOR = TICK_LABEL_COLOR`
- `SUPTITLE_WEIGHT = "bold"`

(Note: `DEFAULT_SUPTITLE_FONT_SIZE` already exists; adding `SUPTITLE_*` names alongside satisfies the scope_brief's mirroring request without breaking the existing constant. Keep both — `DEFAULT_SUPTITLE_FONT_SIZE` is what `figure.py` currently reads.)

Panel base class contract:

```python
class Panel(ABC):
    default_units: ClassVar[Tuple[int, int]] = (1, 1)
    is_text_only: ClassVar[bool] = False  # set True on text-only subclasses

    def __init__(self, *, units: Optional[Tuple[int, int]] = None) -> None:
        self.units = units if units is not None else self.default_units
        ...

    @abstractmethod
    def render(self) -> None: ...
```

TextPanel signature (closest analog):

```python
class TextPanel(Panel):
    is_text_only = True

    def __init__(
        self,
        text: str,
        *,
        units: Optional[Tuple[int, int]] = None,
        font_size: Optional[float] = None,
        color: Optional[str] = None,
        fontweight: str = "bold",
        ...
    ) -> None: ...
```

CompositePanel width-equality guard pattern:

```python
widths = [sum(p.units[0] for p in row) for row in rows]
if any(w != widths[0] for w in widths):
    raise ValueError(f"CompositePanel row widths mismatch: {widths}")
```

Mirror this style for SuptitlePanel in `Figure.compose`.

Heatmap.__init__ today (the only real plottable gap):

```python
def __init__(
    self,
    data: np.ndarray,
    *,
    duration_s: ... = None,
    ...
    zorder: int = 1,
) -> None:
    super().__init__(color=None, linewidth=None, alpha=1.0, ...)
```

Note `alpha` is hardcoded to 1.0 in the super() call and `imshow()` is NOT given alpha. The fix: add `alpha: float = 1.0` to `__init__`, pass to super, and pass `alpha=self.alpha` to `ax.imshow(...)`. Line and TimeSeries already have this — verify only, no edit needed.

figure_1 overlay reference (already working; do not change):

```python
# row1: TimeSeriesPanel with twin_y=True; Line drawn on twin axis via add_twin()
# row2/row3: HeatmapPanel; Line.add() onto primary axis, caller pre-transforms
#            inst_freq_hz → inst_freq_bins via np.interp before passing to Line
row2.add(Heatmap(stft_mag_log, ...))
# (note: no `row2.add(Line(...))` in current figure_1 — Line overlay only on row1
# via twin_y. The scope_brief's mention of "row2/row3 Line-on-heatmap" reflects
# the GENERAL capability HeatmapPanel must support; the documented contract goes
# into HeatmapPanel's docstring regardless of whether figure_1 currently exercises
# it on rows 2/3.)
```

Style invariant references (read-only audit):
- `DEFAULT_PAD_INCHES = 1.0` — the v52 unitary knob
- `DEFAULT_MARGIN_INCHES = DEFAULT_PAD_INCHES`
- `DEFAULT_GUTTER_INCHES = 2.0 * DEFAULT_PAD_INCHES`
- `DEFAULT_COLUMN_GUTTER_INCHES = 2.0 * DEFAULT_PAD_INCHES`
- `DEFAULT_INNER_GUTTER_INCHES = 0.6` (composite-inner; intentionally NOT derived from PAD per existing comment)
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add SuptitlePanel + SUPTITLE_* style constants; route Figure.compose suptitle sugar through it</name>
  <files>
    research/dsplot/panels/suptitle_panel.py (new),
    research/dsplot/panels/__init__.py,
    research/dsplot/__init__.py,
    research/dsplot/style.py,
    research/dsplot/figure.py
  </files>
  <action>
    Create `research/dsplot/panels/suptitle_panel.py` defining `class SuptitlePanel(TextPanel)`. Subclass `TextPanel` (it already renders centered text with `is_text_only=True`, font_size/color/fontweight/auto_shrink — the exact shape needed). The subclass's only purpose is to:
    (a) carry distinct style defaults (SUPTITLE_FONT_SIZE / SUPTITLE_COLOR / SUPTITLE_WEIGHT instead of TextPanel's defaults), and
    (b) act as a discriminable type for the Figure.compose suptitle-routing logic.

    Constructor signature: `def __init__(self, text: str, *, units: Optional[Tuple[int, int]] = None, font_size: Optional[float] = None, color: Optional[str] = None, fontweight: Optional[str] = None, auto_shrink: bool = False) -> None`. Resolve `font_size` to `style.SUPTITLE_FONT_SIZE`, `color` to `style.SUPTITLE_COLOR`, `fontweight` to `style.SUPTITLE_WEIGHT` when None — lazy resolution at construction is fine (do NOT lazy-resolve in `render()` to keep TextPanel's render path unchanged). Default `auto_shrink=False` because the suptitle text is single-line and should NOT be shrunk to fit a narrow cell. Pass everything else through to TextPanel via `super().__init__(text, units=units, font_size=resolved, color=resolved, fontweight=resolved, auto_shrink=auto_shrink)`. No `render()` override needed — TextPanel.render() does the right thing.

    Add to `research/dsplot/style.py` (in the TYPOGRAPHY section, near `DEFAULT_SUPTITLE_FONT_SIZE`):
    ```
    SUPTITLE_FONT_SIZE = DEFAULT_SUPTITLE_FONT_SIZE  # 32
    SUPTITLE_COLOR     = TICK_LABEL_COLOR           # "#888888"
    SUPTITLE_WEIGHT    = "bold"
    ```
    Keep `DEFAULT_SUPTITLE_FONT_SIZE` intact — `figure.py` reads it directly in the legacy suptitle path. The new `SUPTITLE_*` names are siblings, not replacements.

    Wire the sugar in `research/dsplot/figure.py` `Figure.compose`:
    - When `suptitle: str | None` is non-None AND the first row is NOT already a SuptitlePanel-only row: prepend a synthetic row `[SuptitlePanel(text=suptitle, units=(n_cols, 1))]` to `rows` BEFORE the existing width-equality and figsize math runs. `n_cols` is the existing computed value (`width_units_per_row[0]`). This means the existing `_mpl_fig.suptitle(...)` legacy path in `Figure.__init__` must NOT also fire — solve this by passing `suptitle=None` to the `cls(...)` constructor call inside compose (since the suptitle is now a real Panel row).
    - When the user passes an explicit SuptitlePanel-only first row (detect via `isinstance(rows[0][0], SuptitlePanel) and len(rows[0]) == 1`): leave `rows` untouched and skip the legacy `_mpl_fig.suptitle` path the same way.
    - Width-equality guard (mirrors the CompositePanel pattern): the SuptitlePanel row's `units[0]` must equal the sibling rows' total width. The existing width-equality block (`if any(w != width_units_per_row[0] for w in width_units_per_row): raise ValueError(...)`) already enforces this once the SuptitlePanel row is prepended — no extra code needed, but verify the auto-prepend path constructs the SuptitlePanel with `units=(n_cols, 1)` so it passes.
    - Top-reserve math: the legacy code computes `top_reserve = 2.0 * margin` when `suptitle is not None`. With SuptitlePanel now occupying a real grid row at the top, the suptitle's vertical real estate comes from that row's height (one unit tall by default), so the previous `top_reserve = 2.0 * margin` band is NO LONGER needed when the panel-row replaces it. **However**, to preserve bit-identical output for existing callers, do NOT change the legacy `2.0 * margin` reservation — instead, make the SuptitlePanel row use a SHORTER row height that totals exactly `2 * margin` (i.e. `unit_height_inches` for that specific row). The simplest preservation: keep the legacy code path (`_mpl_fig.suptitle(...)` in `Figure.__init__`) untouched and route compose's `suptitle=...` sugar through it as before, AND in parallel allow user-passed explicit SuptitlePanel rows to add their own top row. Concretely:
      * If sugar is used (`suptitle="..."` kwarg, no explicit SuptitlePanel row): keep behavior bit-identical by leaving the existing legacy path alone — do NOT auto-prepend a SuptitlePanel row. Just instantiate SuptitlePanel internally for type-discoverability if a downstream caller wants to introspect it, but render via the legacy suptitle path.
      * If explicit (`rows=[[SuptitlePanel(...)], ...]`): treat it as a real panel row, validate width via the existing equality check, and DO NOT also call `_mpl_fig.suptitle(...)`.

    Translation: Take the SIMPLER preservation route — leave the sugar path unchanged (still calls `_mpl_fig.suptitle`), and only add explicit-SuptitlePanel-row support. The SuptitlePanel class exists so users CAN compose it explicitly when they want it as a real panel; the sugar is unchanged. This bit-identically preserves figure_1, lego_demo, and every existing caller.

    Note in the SuptitlePanel docstring: "Pass via `Figure.compose(suptitle='...')` for the sugar path (legacy rendering via `_mpl_fig.suptitle`), or as an explicit first-row Panel via `rows=[[SuptitlePanel('...')], ...]` to participate in the grid as a real panel."

    Export `SuptitlePanel` from `research/dsplot/panels/__init__.py` (add to `__all__` and the `from .suptitle_panel import SuptitlePanel` line). Export from `research/dsplot/__init__.py` (add to the `from .panels import (...)` block and the top-level `__all__`).

    Imports: in `figure.py`, add the import deferred to inside `compose()` body to avoid module-load cycles: `from .panels.suptitle_panel import SuptitlePanel`. In `suptitle_panel.py`, use `from .text_panel import TextPanel` and `from .. import style`. Use relative imports only; never `from research.X import` (per [[project-dsplot-import-worldviews]]).

    Width-equality guard: when user passes `rows=[[SuptitlePanel(units=(2,1))], [PanelA, PanelB]]` with sibling-row total = 4, the existing `width_units_per_row` equality check fires `ValueError(f"Figure.compose row widths mismatch: {width_units_per_row}")`. That message style already mirrors CompositePanel — no change needed. Verify the error fires by running the verification command.

    DO NOT run `git commit` at any point — per [[feedback-no-auto-commits]] the user controls commit cadence. Stage changes for diff review if helpful, but stop short of commit.
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader &amp;&amp; python -c "
import sys; sys.path.insert(0, 'research')
from dsplot import SuptitlePanel, Figure, TextPanel, HeatmapPanel, style
# Sugar path renders without error
fig1 = Figure.compose(rows=[[HeatmapPanel(), HeatmapPanel()]], suptitle='sugar test')
assert fig1._has_suptitle is True
fig1.close()
# Explicit path: width-matched row OK
fig2 = Figure.compose(rows=[[SuptitlePanel('explicit', units=(2,1))], [HeatmapPanel(), HeatmapPanel()]])
fig2.close()
# Explicit path: width-mismatched row raises ValueError
try:
    Figure.compose(rows=[[SuptitlePanel('bad', units=(2,1))], [HeatmapPanel(), HeatmapPanel(), HeatmapPanel()]])
    raise AssertionError('expected ValueError for width mismatch')
except ValueError as e:
    assert 'mismatch' in str(e), f'unexpected error message: {e}'
# Style constants exist with expected values
assert style.SUPTITLE_FONT_SIZE == style.DEFAULT_SUPTITLE_FONT_SIZE == 32
assert style.SUPTITLE_COLOR == style.TICK_LABEL_COLOR == '#888888'
assert style.SUPTITLE_WEIGHT == 'bold'
print('Task 1 OK')
"</automated>
  </verify>
  <done>
    SuptitlePanel exists and is exported from both `research.dsplot.panels` and `research.dsplot`. SUPTITLE_FONT_SIZE / SUPTITLE_COLOR / SUPTITLE_WEIGHT exist in style.py mirroring the existing render values. `Figure.compose(suptitle='...')` sugar path is unchanged (bit-identical legacy rendering). Explicit `rows=[[SuptitlePanel(...)], ...]` composition works AND triggers the existing width-equality ValueError on mismatch. No `git commit` invoked.
  </done>
</task>

<task type="auto">
  <name>Task 2: Cross-type plottable overlay contract — add Heatmap alpha kwarg, verify Line/TimeSeries already pass zorder + alpha, document HeatmapPanel overlay contract</name>
  <files>
    research/dsplot/plottables/heatmap.py,
    research/dsplot/panels/heatmap_panel.py
  </files>
  <action>
    Update `research/dsplot/plottables/heatmap.py`:
    - Add `alpha: float = 1.0` to `Heatmap.__init__` (placed alongside `vmin`/`vmax`/`zorder` for kwarg grouping).
    - Replace the hardcoded `alpha=1.0` in the `super().__init__(...)` call with `alpha=alpha` (pass-through).
    - In `draw(self, ax)`, add `alpha=self.alpha` to the `ax.imshow(...)` call (place alongside vmin/vmax/zorder).
    - Default zorder remains `1` per existing code — matches the scope_brief target (Heatmap=1, TimeSeries=2, Line=3).
    - No other changes to Heatmap.

    Verify (read-only, no edits) that `Line` and `TimeSeries` already accept `zorder` + `alpha` and pass them through. Confirmed during planning:
    - `Line.__init__` takes `alpha=1.0, zorder=3`; `draw()` passes both to `ax.plot(...)`.
    - `TimeSeries.__init__` takes `alpha=1.0, zorder=2`; `draw()` passes both to `ax.fill_between(...)`.
    No edits needed to these two files — but the task's verify step will assert the contract anyway.

    Update `research/dsplot/panels/heatmap_panel.py` module docstring to add a paragraph (after the existing module docstring, before any class definition) documenting the Line-on-Heatmap overlay contract:

    "**Line overlays on HeatmapPanel:** A `Line` plottable added via `panel.add(Line(...))` draws onto the panel's primary axis, which is in HEATMAP COORDINATE SPACE — i.e. x is duration (seconds, or whatever `Heatmap.extent`'s x-range is) and y is BIN INDEX (0 to `len(freqs)`), not Hz. Callers overlaying a frequency-domain curve (e.g. instantaneous frequency in Hz) must pre-transform the y-values into bin-space before constructing the Line, typically via `np.interp(inst_freq_hz, freqs, np.arange(len(freqs)))`. HeatmapPanel does NOT support a twin y-axis — overlays share the primary axis only. See `dsplot/figures/figure_1.py::_build_3row_figure` for the canonical pattern (twin-axis Line on row 1's TimeSeriesPanel, primary-axis Line overlays available on rows 2/3's HeatmapPanels via the bin-space transform)."

    The docstring goes at the MODULE level (top of file), not on the class — but if you'd rather put it on the class, that's fine too. Keep it as one paragraph; do not introduce a new section heading.

    DO NOT run `git commit`. Per [[feedback-no-auto-commits]] the user controls commit cadence.
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader &amp;&amp; python -c "
import sys; sys.path.insert(0, 'research')
import numpy as np
from dsplot import Heatmap, Line, TimeSeries, HeatmapPanel
# Heatmap alpha kwarg
hm = Heatmap(np.zeros((4, 4)), alpha=0.5)
assert hm.alpha == 0.5, f'Heatmap.alpha not stored: {hm.alpha}'
hm_default = Heatmap(np.zeros((4, 4)))
assert hm_default.alpha == 1.0, f'Heatmap default alpha wrong: {hm_default.alpha}'
assert hm_default.zorder == 1, f'Heatmap default zorder wrong: {hm_default.zorder}'
# Line zorder + alpha
ln = Line(np.arange(4), np.arange(4), alpha=0.7, zorder=5)
assert ln.alpha == 0.7 and ln.zorder == 5
ln_d = Line(np.arange(4), np.arange(4))
assert ln_d.alpha == 1.0 and ln_d.zorder == 3
# TimeSeries zorder + alpha
ts = TimeSeries(np.zeros(8), 1000.0, alpha=0.6, zorder=4)
assert ts.alpha == 0.6 and ts.zorder == 4
ts_d = TimeSeries(np.zeros(8), 1000.0)
assert ts_d.alpha == 1.0 and ts_d.zorder == 2
# HeatmapPanel docstring contract
import inspect
from dsplot.panels import heatmap_panel
doc = (heatmap_panel.__doc__ or '') + (HeatmapPanel.__doc__ or '')
assert 'primary axis' in doc.lower(), 'HeatmapPanel overlay contract missing from docstring'
print('Task 2 OK')
"
# Worldview B smoke
python -c "from research.dsplot import Heatmap, Line, TimeSeries; print('B ok')"</automated>
  </verify>
  <done>
    Heatmap accepts `alpha` kwarg and passes it to `imshow()`. Line/TimeSeries verified to already accept zorder+alpha. HeatmapPanel docstring documents the bin-space overlay contract. Both import worldviews pass smoke tests. No `git commit` invoked.
  </done>
</task>

<task type="auto">
  <name>Task 3: Pad invariant audit (read-only) + render figure_1 hero for visual smoke test + run both worldview smoke tests</name>
  <files>
    .planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md (new)
  </files>
  <action>
    Read-only audit. DO NOT edit any panel/style code in this task — surface findings in SUMMARY.md.

    1. Grep audit across `research/dsplot/panels/*.py` and `research/dsplot/figure.py` for hardcoded pad/margin/inches/gutter values that bypass the v52 unitary knob (`style.DEFAULT_PAD_INCHES`, `style.DEFAULT_MARGIN_INCHES`, `style.DEFAULT_GUTTER_INCHES`, `style.DEFAULT_COLUMN_GUTTER_INCHES`, `style.DEFAULT_INNER_GUTTER_INCHES`).

       Suggested grep:
       ```
       grep -nE "[0-9]+\.[0-9]+|[0-9]+" research/dsplot/panels/*.py research/dsplot/figure.py \
         | grep -iE "pad|margin|inches|gutter|inset" \
         | grep -v "from\|import" \
         | grep -v "style\."
       ```

       For each hit, classify:
       - DERIVED (uses a style.* constant — OK)
       - INTENTIONALLY-CUSTOM (e.g. `DEFAULT_INNER_GUTTER_INCHES = 0.6` in style.py is intentionally separate from the unitary knob per existing comment; `text_panel.cell_padding_frac=0.08` is panel-local fractional padding, not figure-level pad) — OK, document why.
       - BYPASS (hardcoded number where a style constant should be used) — RECORD in SUMMARY.md "Pad audit findings".

       Expected bypasses: NONE based on planner's pre-read. Common hits that look bypassy but are not:
       - `text_panel.py: cell_padding_frac=0.08` — fractional cell-inset, not inch-domain pad. Document as intentional.
       - `base.py: _base_bottom_pad = 0.18` (InteractivePanel) — control-bar reservation in axes-fraction units, not inches. Document as intentional.
       - `figure.py: 1.05 / fig_h`, `0.75 / fig_h` — figure_number / figure_caption text positions in figure-fraction units, derived from inch-domain reasoning but expressed as numerators. Acceptable for now — flag as "could be parameterized" if you find them.

       DO NOT fix any bypass unless it's trivially a missing `style.` lookup (e.g. literal `1.0` where `style.DEFAULT_PAD_INCHES` is meant) AND fixing it preserves bit-identical output. If anything non-trivial is found: surface in SUMMARY.md under "Pad audit findings" and STOP — do not fix.

    2. Render figure_1 hero for visual smoke test:
       ```
       cd /home/eddie-water/dev/python/sub-shader
       python -c "
       import sys; sys.path.insert(0, 'research')
       from dsplot.figures.figure_1 import render_hero
       path = render_hero()
       print('hero rendered to:', path)
       "
       ```
       Confirm the PNG was generated without errors (subprocess exits 0; printed path exists). DO NOT pixel-diff — the human checkpoint in Task 4 handles visual verification.

    3. Run both import worldviews smoke tests (per [[project-dsplot-import-worldviews]]):
       ```
       python -c "import sys; sys.path.insert(0, 'research'); from dsplot.figures import figure_1, figure_2_4_1; print('A ok')"
       python -c "from research.dsplot.figures.figure_1 import render_hero; print('B ok')"
       ```
       Both must print their ok line.

    4. Write `.planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md` using the standard summary template, with sections:
       - "What changed" — bulleted file list with one-line description each (suptitle_panel.py NEW, panels/__init__.py + dsplot/__init__.py exports, style.py SUPTITLE_* constants, figure.py SuptitlePanel routing, heatmap.py alpha kwarg, heatmap_panel.py docstring).
       - "Pad audit findings" — table of every grep hit with classification (DERIVED / INTENTIONALLY-CUSTOM / BYPASS). State explicitly if no bypasses found.
       - "Smoke tests" — paste the worldview A + B output lines + the hero render path.
       - "Deferred / open questions" — note that the sugar path was preserved bit-identically by leaving the legacy `_mpl_fig.suptitle()` call alone (i.e. SuptitlePanel sugar does NOT route through SuptitlePanel.render() — only explicit composition does). If the user wants TRUE unification (sugar also rendered via SuptitlePanel.render()) that's a follow-on requiring layout-math changes.
       - "Files NOT committed" — proposed commit message + file list the user can copy when they decide to commit. Surface this per [[feedback-no-auto-commits]].

    DO NOT run `git commit`.
  </action>
  <verify>
    <automated>cd /home/eddie-water/dev/python/sub-shader &amp;&amp; python -c "
import sys; sys.path.insert(0, 'research')
from dsplot.figures import figure_1, figure_2_4_1
print('A ok')
" &amp;&amp; python -c "from research.dsplot.figures.figure_1 import render_hero; print('B ok')" &amp;&amp; python -c "
import sys; sys.path.insert(0, 'research')
from dsplot.figures.figure_1 import render_hero
path = render_hero()
import os
assert os.path.exists(path), f'hero PNG not found: {path}'
print('hero rendered ok:', path)
" &amp;&amp; test -f .planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md &amp;&amp; grep -q "Pad audit findings" .planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md &amp;&amp; echo "Task 3 OK"</automated>
  </verify>
  <done>
    Pad audit findings recorded in SUMMARY.md. Both worldview smoke tests pass. figure_1.render_hero() produces a PNG without error. SUMMARY.md includes "What changed", "Pad audit findings", "Smoke tests", "Deferred", and "Files NOT committed" sections.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Human visual verification — re-render figure_1 hero, confirm no visual regression</name>
  <what-built>
    - SuptitlePanel + SUPTITLE_* style constants (sugar path bit-identically preserved)
    - Heatmap alpha kwarg (Line/TimeSeries already correct)
    - HeatmapPanel overlay-contract docstring
    - Pad invariant audit findings written to SUMMARY.md
    - figure_1 hero re-rendered as smoke test
  </what-built>
  <how-to-verify>
    1. The executor will have written the hero PNG path on stdout (something like `assets/images/dsp/figures/figure_1/fourier_vs_wavelet.png`). Open that file.
    2. Compare against the prior locked hero render (v8 reference referenced in [[project-figure-1]]). Specifically check:
       - Row 1: time series + orange inst-freq curve on the TWIN (left) axis, with Hz tick labels on the left and the orange line backed by a dark outline. Bit-identical to before.
       - Row 2: STFT spectrogram with log-spaced Hz y-ticks on the LEFT (primary axis). No Line overlay on this row in current figure_1 (overlay capability exists; figure_1 just uses twin-axis on row 1).
       - Row 3: CWT spectrogram, same chrome as row 2.
       - Suptitle ("Fourier vs Wavelet Decomposition") renders at the top in the same size/weight/color as before (it should — sugar path is untouched).
       - Figure number "Figure 1" + caption render at the bottom, unchanged.
       - Cell borders, gutters, margins look unchanged.
    3. Optionally re-render `lego_demo` to confirm its suptitle "dsplot — Sample Template" still renders identically:
       ```
       python -c "
       import sys; sys.path.insert(0, 'research')
       from dsplot.figures.lego_demo import build_figure
       fig = build_figure(); fig.render()
       fig.savefig('assets/images/dsp/lego_demo_svu_smoke.png')
       "
       ```
       Open `assets/images/dsp/lego_demo_svu_smoke.png` and compare to the most recent `lego_demo_v*.png` in `assets/images/dsp/`. Suptitle styling should match.
    4. Skim SUMMARY.md "Pad audit findings" — confirm the table makes sense and there are no surprise bypasses.
    5. Per [[feedback-no-auto-commits]]: NOTHING is committed. SUMMARY.md "Files NOT committed" section lists what the executor would have committed; you decide if/when/how to commit.
  </how-to-verify>
  <resume-signal>
    Type "approved" if the figure_1 hero is visually unchanged AND the pad audit findings look sane. Otherwise describe the regression (e.g. "row 1 inst-freq curve color shifted" or "suptitle font size changed") and the executor will iterate.
  </resume-signal>
</task>

</tasks>

<verification>
- Both import worldviews resolve cleanly (notebook + repo-root).
- `Figure.compose(suptitle='...', rows=[...])` renders bit-identical to before (legacy sugar path untouched).
- `Figure.compose(rows=[[SuptitlePanel('...', units=(N,1))], [...]])` renders an equivalent suptitle when explicitly composed.
- Width-equality guard fires on mismatched SuptitlePanel-row width with a ValueError mirroring CompositePanel's message style.
- Heatmap accepts `alpha=` and passes it to `imshow`; Line/TimeSeries already pass zorder+alpha (verified).
- HeatmapPanel docstring documents the bin-space overlay contract.
- figure_1.render_hero() renders without error; visually unchanged (human checkpoint).
- Pad audit findings recorded in SUMMARY.md; no silent bypasses introduced.
- No `git commit` was run by the executor (per [[feedback-no-auto-commits]]).
</verification>

<success_criteria>
- All four tasks' `<verify>` blocks pass (Tasks 1–3 automated; Task 4 = user approves).
- SuptitlePanel is importable from both `dsplot` and `research.dsplot`.
- SUPTITLE_FONT_SIZE / SUPTITLE_COLOR / SUPTITLE_WEIGHT exist in `style.py` with the values that preserve current rendering.
- Heatmap accepts `alpha=`; HeatmapPanel module docstring documents the overlay contract.
- Pad audit findings table exists in SUMMARY.md with each grep hit classified.
- figure_1 hero PNG generated; user confirms visual parity in Task 4.
- Zero git commits made by the executor; SUMMARY.md "Files NOT committed" lists what the user can review and commit themselves.
</success_criteria>

<output>
Create `.planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md` when Tasks 1-3 complete. Task 4 (human checkpoint) blocks final completion until user approves.
</output>
