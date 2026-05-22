---
phase: quick-260521-svu
plan: 01
subsystem: research/dsplot
tags: [dsplot, panels, plottables, suptitle, overlays, refactor]
status: awaiting-human-verify
requirements:
  - SVU-G1-suptitle-panel
  - SVU-G2-cross-type-overlays
  - SVU-G3-pad-audit
key_files:
  created:
    - research/dsplot/panels/suptitle_panel.py
  modified:
    - research/dsplot/panels/__init__.py
    - research/dsplot/__init__.py
    - research/dsplot/style.py
    - research/dsplot/panels/heatmap_panel.py
    - research/dsplot/plottables/heatmap.py
metrics:
  tasks_completed: 3
  human_checkpoint_pending: true
  files_created: 1
  files_modified: 5
  git_commits_by_executor: 0
---

# Quick 260521-svu Summary

SuptitlePanel + cross-type plottable overlays + pad-invariant audit.

## What changed

- **NEW** `research/dsplot/panels/suptitle_panel.py` — `class SuptitlePanel(TextPanel)`. Subclass exists for type-discoverability + distinct style defaults (SUPTITLE_FONT_SIZE / SUPTITLE_COLOR / SUPTITLE_WEIGHT) and `auto_shrink=False` default. Renders via inherited `TextPanel.render()`; no override.
- **MODIFIED** `research/dsplot/panels/__init__.py` — adds `from .suptitle_panel import SuptitlePanel` + `__all__` entry.
- **MODIFIED** `research/dsplot/__init__.py` — re-exports `SuptitlePanel` at the top level alongside `TextPanel`, etc.
- **MODIFIED** `research/dsplot/style.py` — adds `SUPTITLE_FONT_SIZE = DEFAULT_SUPTITLE_FONT_SIZE (32)`, `SUPTITLE_COLOR = TICK_LABEL_COLOR ("#888888")`, `SUPTITLE_WEIGHT = "bold"`. Mirrors values the legacy `_mpl_fig.suptitle(...)` path already uses → bit-identical preservation. `DEFAULT_SUPTITLE_FONT_SIZE` left intact (still read by `figure.py`).
- **MODIFIED** `research/dsplot/plottables/heatmap.py` — adds `alpha: float = 1.0` kwarg to `Heatmap.__init__`, passes it to `super().__init__(alpha=alpha, ...)` (instead of the prior hardcoded `1.0`), and to `ax.imshow(..., alpha=self.alpha, ...)`. Line and TimeSeries already had this; no edits needed there.
- **MODIFIED** `research/dsplot/panels/heatmap_panel.py` — module docstring extended with "Line overlays on HeatmapPanel" paragraph documenting the bin-space overlay contract (x = duration, y = bin index, NOT Hz; no twin y-axis; pre-transform Hz → bins via `np.interp`).
- **NOT MODIFIED** `research/dsplot/figure.py` — per the plan's simpler-preservation route. The sugar path `Figure.compose(suptitle="…")` continues to route through the legacy `_mpl_fig.suptitle(...)` call. SuptitlePanel is available for explicit composition via `rows=[[SuptitlePanel("…", units=(N,1))], …]`; the existing width-equality check in `Figure.compose` validates the row width without needing new code.

## Verify steps (Tasks 1 + 2 automated)

```
Task 1 OK    # SuptitlePanel exported both worldviews; sugar+explicit paths work; width-mismatch ValueError fires; SUPTITLE_* constants exist
Task 2 OK    # Heatmap.alpha kwarg + imshow pass-through; Line/TimeSeries verified to already accept zorder+alpha; HeatmapPanel docstring contains "primary axis"
B ok         # Worldview B: from research.dsplot import Heatmap, Line, TimeSeries
```

## Smoke tests (Task 3)

```
A ok                                                                   # sys.path-shim notebook style
B ok                                                                   # research.dsplot repo-root style
hero rendered to: /home/eddie-water/dev/python/sub-shader/assets/images/dsp/figures/figure_1/hero_click_plus_tone_v1.png
```

The hero PNG was regenerated without errors. Note the actual hero filename is `hero_click_plus_tone_v1.png`, not the `fourier_vs_wavelet.png` referenced in PLAN.md (the plan's filename was a guess from prior memory; the locked v8 hero output path is what `figure_1.render_hero()` writes today).

## Pad audit findings

Grep target: `research/dsplot/panels/*.py` + `research/dsplot/figure.py` for hardcoded `pad|margin|inches|gutter|inset` literals that bypass `style.*` constants.

| File:line                            | Hit                                          | Classification          | Notes                                                                                                                                                            |
| ------------------------------------ | -------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| panels/time_series_panel.py:157, 162 | `fig_w_in = ...get_size_inches()[0]`, `1.0 + half_pad_in / axes_w_in` | DERIVED                | Twin-axis label placement; `half_pad_in = style.DEFAULT_AXIS_LABEL_INSET_INCHES` upstream — derived from style.                                                  |
| panels/interactive_panel.py:45       | `_base_bottom_pad: float = 0.18`             | INTENTIONALLY-CUSTOM    | Axes-fraction reservation for InteractivePanel's prev/next button bar — not inch-domain pad. Distinct concept from the unitary PAD knob.                         |
| panels/interactive_panel.py:184      | `borderpad=0`                                | INTENTIONALLY-CUSTOM    | matplotlib `AnchoredOffsetbox.borderpad` (different `pad` namespace) — control-bar internal layout.                                                              |
| panels/base.py:46-49                 | `_base_bottom_pad: float = 0.0`              | INTENTIONALLY-CUSTOM    | Abstract default — InteractivePanel overrides. Axes-fraction units, not inches.                                                                                  |
| panels/text_panel.py:37, 108-109     | `cell_padding_frac: float = 0.08`            | INTENTIONALLY-CUSTOM    | Per-text-cell fractional inset for auto-shrink fit math — fractional axes-coord padding, NOT inch-domain pad. Documented as a separate knob from the unitary PAD. |
| figure.py:147, 286, 304, 403         | `top_reserve = 2.0 * margin` / "2*margin" comments | DERIVED                | `margin` resolves from `style.DEFAULT_MARGIN_INCHES` (= `DEFAULT_PAD_INCHES`). The `2.0 *` multiplier is the canonical "1 PAD band + 1 PAD gap" recipe. OK.       |
| figure.py:186                        | `1.05 / fig_h` (figure_number y-position)    | COULD-BE-PARAMETERIZED  | Hardcoded inch-numerator for figure_number text y-position. Inch-domain reasoning but not pulled from `style.*`. Flag — not a fix in this task per plan scope.   |
| figure.py:201                        | `0.75 / fig_h` (figure_caption y-position)   | COULD-BE-PARAMETERIZED  | Same as above for figure_caption. Flag.                                                                                                                          |
| figure.py:294                        | `bottom_reserve = 1.5`                       | COULD-BE-PARAMETERIZED  | Hardcoded inch reservation when `figure_number` or `figure_caption` is set. Could be `1.5 * style.DEFAULT_PAD_INCHES` or a new `DEFAULT_FIGURE_BOTTOM_RESERVE`. Flag. |
| figure.py:413                        | `max_bottom_pad + 0.05`                      | COULD-BE-PARAMETERIZED  | Subtitle-bottom safety margin added on top of panel bottom_pad. Small absolute fudge in figure-fraction units; could parameterize. Flag.                          |
| figure.py:419 (right margin)         | `1.0 - margin_h_frac`                        | DERIVED                | OK — `margin_h_frac` derived from `style.DEFAULT_MARGIN_INCHES`.                                                                                                  |

**Summary:**
- **No BYPASS findings** that silently undermine the v52 unitary PAD knob.
- **4 COULD-BE-PARAMETERIZED** spots in `figure.py` related to figure_number/caption bottom-band layout. These are inch-domain reasoning baked into numerators (`1.05`, `0.75`, `1.5`, `0.05`). Tightly grouped — would make a clean follow-on quick task: introduce `DEFAULT_FIGURE_NUMBER_Y_INCHES`, `DEFAULT_FIGURE_CAPTION_Y_INCHES`, `DEFAULT_FIGURE_BOTTOM_RESERVE_INCHES`, `DEFAULT_SUBTITLE_RESERVE_SAFETY` (or similar). **Not fixed here per the plan's "do not fix non-trivial bypasses, surface only" rule.**

## Deviations from Plan

None. The plan's "simpler-preservation route" was followed verbatim: `figure.py` is unchanged, SuptitlePanel is available for explicit composition only. Sugar path remains bit-identical.

## Deferred / open questions

- The SuptitlePanel sugar path does NOT route through `SuptitlePanel.render()` — only explicit composition does. If true unification (sugar also rendered via `SuptitlePanel.render()`) is desired, that's a follow-on requiring layout-math changes (`top_reserve` math would need to account for a real first-row panel instead of the legacy `2 * margin` band reservation, and the `_mpl_fig.suptitle()` legacy call would need to be removed). Out of scope for this quick task per the plan.
- 4 figure_number/caption layout knobs could be parameterized in `figure.py` (see pad audit table above) — clean follow-on quick task.
- HeatmapPanel currently has no twin-y support (TimeSeriesPanel does, via `add_twin()`). figure_1 row 1 uses twin_y for the inst-freq Hz overlay; if a future figure wants Hz-domain overlays on a Heatmap row, either (a) add twin-y to HeatmapPanel or (b) keep the documented bin-space pre-transform. Tracked but not actioned.

## Files NOT committed

Per [[feedback-no-auto-commits]] this executor made **zero** git commits. Working-tree changes the user can review and commit at their discretion:

```
M research/dsplot/__init__.py
M research/dsplot/panels/__init__.py
M research/dsplot/panels/heatmap_panel.py
M research/dsplot/plottables/heatmap.py
M research/dsplot/style.py
?? research/dsplot/panels/suptitle_panel.py
?? .planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-PLAN.md
?? .planning/quick/260521-svu-suptitlepanel-and-cross-type-plottable-o/260521-svu-SUMMARY.md
```

Suggested commit message when the user is ready (single commit covering all of Q2 svu):

```
feat(dsplot): SuptitlePanel + Heatmap alpha + HeatmapPanel overlay contract

- New SuptitlePanel(TextPanel) subclass exported from dsplot/panels.
- SUPTITLE_FONT_SIZE / SUPTITLE_COLOR / SUPTITLE_WEIGHT in style.py
  (mirror the existing legacy _mpl_fig.suptitle render values).
- Figure.compose(suptitle=...) sugar path preserved bit-identically;
  explicit rows=[[SuptitlePanel(...)], ...] supported via existing
  width-equality guard.
- Heatmap accepts alpha= kwarg and passes it through to ax.imshow().
- HeatmapPanel module docstring documents the bin-space Line-overlay
  contract (x=duration, y=bin index; no twin y-axis).
- Pad-invariant audit: no bypasses found; 4 figure_number/caption
  layout knobs flagged as could-be-parameterized (follow-on).
```

(Or split as separate commits for SuptitlePanel vs Heatmap-alpha vs HeatmapPanel-docstring if preferred.)

## Self-Check

- [x] `research/dsplot/panels/suptitle_panel.py` exists (NEW)
- [x] `SuptitlePanel` importable from `dsplot` (worldview A) and `research.dsplot` (worldview B)
- [x] `style.SUPTITLE_FONT_SIZE == 32`, `style.SUPTITLE_COLOR == "#888888"`, `style.SUPTITLE_WEIGHT == "bold"`
- [x] `Heatmap(data, alpha=0.5).alpha == 0.5`; default still `1.0`; zorder default still `1`
- [x] HeatmapPanel docstring contains "primary axis"
- [x] `figure_1.render_hero()` returns a path; PNG exists at that path
- [x] Both worldview A + B smoke tests print "A ok" / "B ok"
- [x] `git log --oneline -3` head is `e7be375` — no executor commits

**Self-Check: PASSED.** Task 4 (human-verify checkpoint) pending — user opens the regenerated hero PNG, compares against prior v8, and approves or describes any regression.
