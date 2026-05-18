---
phase: 09-dsplot-library-architecture
plan: 05
type: summary
status: pending-visual-verification
subsystem: dsplot/figures (consumer code) + StaticPanel3D (library)
tags: [dsplot, figures, motivator, vector-projection-3d, components-recombine, projection-symmetry, alignment-diagnostic, polymorphic-vector, static-panel-3d, twin-axis-escape-hatch, orphan-retire]
requirements: [PLOT-18, PLOT-21, PLOT-24]
locked_decisions:
  - D-01: library isolation honored — research/dsplot/ proper has zero `from utilities` / `from subshader` imports; figures/ subpackage is the only consumer that bridges to research.utilities
  - D-02: polymorphic Vector with 3-tuples used for vector_projection_3d (no Vector3D class)
  - D-03: vector_xy_reconstruction.png orphan retired — components_recombine.render_vector_xy_reconstruction is the new generator
  - D-04: Figure.add_panel(projection="3d") used for the 3D cell in vector_projection_3d.py
  - D-05: motivator-local layout constants derive from dsplot.style.DEFAULT_* (LAYOUT_HSPACE, LAYOUT_LABEL_RATIO, LAYOUT_MARGIN, ROW_LABEL_FONT_SIZE, AXIS_LABEL_FONT_SIZE, TICK_LABEL_FONT_SIZE)
key-files:
  created:
    - research/dsplot/figures/__init__.py
    - research/dsplot/figures/__main__.py
    - research/dsplot/figures/foundation_constants.py
    - research/dsplot/figures/vector_basics.py
    - research/dsplot/figures/dot_product_geometry.py
    - research/dsplot/figures/components_recombine.py
    - research/dsplot/figures/projection_reconstruction.py
    - research/dsplot/figures/vector_projection_3d.py
    - research/dsplot/figures/motivator.py
    - research/dsplot/figures/alignment_diagnostic.py
    - research/dsplot/panels/static_panel_3d.py
  modified:
    - research/dsplot/figure.py  (subplots_adjust top=0.88 when suptitle present; canvas.layout guard tightened to require hasattr "width")
    - research/dsplot/__init__.py  (export StaticPanel3D)
    - research/dsplot/panels/__init__.py  (export StaticPanel3D)
    - research/dsplot/panels/static_panel.py  (title y=1.07 / subtitle y=1.00 stacking when both present)
    - research/dsplot/plottables/annotation.py  (Axes3D branch uses ax.text2D + transAxes)
    - research/tests/dsplot/test_panel_composition.py  (assertion updated to y=1.00)
generated-figures:
  dsp:
    - assets/images/dsp/vector_basics_09_05.png
    - assets/images/dsp/dot_product_geometry_09_05.png
    - assets/images/dsp/components_recombine_either_order_v18_09_05.png
    - assets/images/dsp/vector_xy_reconstruction_09_05.png  (D-03 orphan retire)
    - assets/images/dsp/projection_reconstruction_either_order_v9_09_05.png
    - assets/images/dsp/vector_projection_3d_v2_combo5_palette_09_05.png
  generated:
    - assets/images/generated/dsp_motivator_v4_100-2000hz_0.5s_09_05.png
    - assets/images/generated/dsp_motivator_v5_50-5000hz_1.0s_09_05.png
    - assets/images/generated/dsp_motivator_vw1_gentle_100-2000hz_0.5s_09_05.png
    - assets/images/generated/dsp_motivator_vw2_moderate_100-2000hz_0.5s_09_05.png
    - assets/images/generated/dsp_motivator_vw3_aggressive_100-2000hz_0.5s_09_05.png
    - assets/images/generated/dsp_motivator_vw4_aggressive_20-20000hz_2.0s_09_05.png
    - assets/images/generated/dsp_alignment_diagnostic_09_05.png
commits:
  - 481d83b: feat(09-05) port simple-vector figures + foundation constants + dispatcher
  - 75a9867: feat(09-05) port projection_reconstruction + 3D foundation + StaticPanel3D
  - 25413db: feat(09-05) port motivator (6 versions) + alignment_diagnostic
---

# 09-05 Summary — Port DSP.md figures to dsplot

Every figure referenced from `src/subshader/dsp/DSP.md` now has a dsplot-based generator under `research/dsplot/figures/`. The canonical `research/dsp_figures.py` is now obsoleted by the new dispatcher (`python -m dsplot.figures`), which writes either to `assets/images/dsp/` (foundation diagrams) or `assets/images/generated/` (motivator/alignment hero figures). Filenames match the originals exactly — DSP.md alt-text resolves unchanged after the merge-time rename strips `_09_05`.

## Per-figure module map

| Original PNG | New module | Generator |
| --- | --- | --- |
| `assets/images/dsp/vector_basics.png` | `research/dsplot/figures/vector_basics.py` | `render(out_dir, filename)` |
| `assets/images/dsp/dot_product_geometry.png` | `research/dsplot/figures/dot_product_geometry.py` | `render(out_dir, filename)` |
| `assets/images/dsp/components_recombine_either_order_v18.png` | `research/dsplot/figures/components_recombine.py` | `render(out_dir, filename)` |
| `assets/images/dsp/vector_xy_reconstruction.png` (D-03 orphan retire) | `research/dsplot/figures/components_recombine.py` | `render_vector_xy_reconstruction(out_dir, filename)` |
| `assets/images/dsp/projection_reconstruction_either_order_v9.png` | `research/dsplot/figures/projection_reconstruction.py` | `render(out_dir, filename)` |
| `assets/images/dsp/vector_projection_3d_v2_combo5_palette.png` | `research/dsplot/figures/vector_projection_3d.py` | `render(out_dir, filename)` |
| `assets/images/generated/dsp_motivator_*.png` (6 versions) | `research/dsplot/figures/motivator.py` | `render_all(out_dir, suffix)` over `VERSIONS` |
| `assets/images/generated/dsp_alignment_diagnostic.png` | `research/dsplot/figures/alignment_diagnostic.py` | `render(out_dir, filename)` |

## Decisions honored

**D-01 (library isolation)** — verified via:

```
grep -rIn "^from utilities\|^from subshader\|^import utilities\|^import subshader" \
  research/dsplot/ --exclude-dir=figures
# → zero matches
```

The library proper (everything under `research/dsplot/` except `figures/`) has no imports from `research.utilities` or `src/subshader`. Only `figures/motivator.py` and `figures/alignment_diagnostic.py` bridge to `utilities` (for `compute_full_cwt`, `plot_*_spectrogram`, `build_*_chirp`, `create_grid_scaffold`). This is consumer code — the bridge is explicitly allowed by D-01.

**D-02 (polymorphic Vector, no Vector3D)** — `vector_projection_3d.py` builds 7 Vector instances total (1 bold `a` + 6 reconstruction segments), every one constructed with a 3-tuple `(x, y, z)`. The same `Vector` class draws 2D arrows in `vector_basics.py` / `dot_product_geometry.py` / `projection_reconstruction.py` from 2-tuples. The 3-tuple → 3D dispatch happens inside `Vector.draw(ax)` by inspecting the axes type, not by class identity.

**D-03 (orphan retire)** — `vector_xy_reconstruction.png` was historically a frozen render with no live generator. `components_recombine.render_vector_xy_reconstruction` is the new generator: a 2-panel figure rendering `a = (cos π/6, sin π/6)` decomposed tip-to-tail in opposite orders (x→y vs y→x). The dispatcher includes it in the default render set.

**D-04 (Figure.add_panel(projection="3d"))** — `vector_projection_3d.py` creates its 3D cell via `fig.add_panel(static_panel_3d, row=0, col=0, projection="3d")`. The `projection` kwarg flows through `Figure._cells` → `subplot_kw["projection"]` so matplotlib creates the Axes3D backing.

**D-05 (lazy style template + local overrides)** — `motivator.py` declares module-level constants:

```python
LAYOUT_HSPACE       = style.DEFAULT_HSPACE
LAYOUT_LABEL_RATIO  = style.DEFAULT_LABEL_RATIO
LAYOUT_MARGIN       = style.DEFAULT_PANEL_MARGIN
ROW_LABEL_FONT_SIZE = style.DEFAULT_ROW_LABEL_SIZE
AXIS_LABEL_FONT_SIZE = style.DEFAULT_AXIS_LABEL_SIZE
TICK_LABEL_FONT_SIZE = style.DEFAULT_TICK_LABEL_SIZE
```

No hardcoded numerics. Override globally by reassigning `dsplot.style.DEFAULT_*`; override motivator-only by editing those six module-level constants. Other figure modules use `style.DEFAULT_PANEL_SIZE_INCHES * <multiplier>` for figsize, also derived.

**D-07 (matplotlib.widgets)** — not exercised by 09-05 (no InteractivePanel-based figures in this set).

## Key architectural notes

### StaticPanel3D — the 3D panel chrome subclass

`research/dsplot/panels/static_panel_3d.py` is the new 3D-specific Panel. It exists because:

- 3D axis chrome (spines, x/y/z labels at tips, view_init, lim cube) is genuinely different from 2D arrow chrome and doesn't share with `StaticPanel`.
- 3D Vector arrows need `computed_zorder=False` so the host axes respects each Plottable's `zorder` — `StaticPanel3D.render()` sets that flag.
- The 6 neutral axis-spine arrows on `StaticPanel3D` are themselves polymorphic `Vector` instances with 3-tuples (`±x, ±y, ±z`), reinforcing D-02 from the inside: the chrome uses the same Plottable that consumer code uses.

### Twin-axis matplotlib escape hatch (motivator row 0)

The motivator's top row pairs a grayscale time-series with the yellow inst-freq curve on a SHARED x-axis but INDEPENDENT y-scales. That doesn't fit the Plottable contract (single Axes per Plottable). Rather than force a `TimeSeriesWithTwinAxis` Plottable that exists only for one figure, `motivator.py` uses `ax.twinx()` directly. This is the documented "escape hatch" — `motivator.py`'s docstring labels it explicitly so readers know exactly which line crosses the abstraction boundary.

### Foundation constants module

`research/dsplot/figures/foundation_constants.py` houses the four shared vector constants (`A = (2.0, 3.0)`, `A_PRIME = (-2.0, 3.0)`, `B = (3.0, 2.0)`, `A_Z = 1.5`, `FOUND_LIM = 6`). Three figures (`vector_basics`, `dot_product_geometry`, `projection_reconstruction`, `components_recombine`, `vector_projection_3d`) import from this module — the canonical visual identity stays consistent across panels, and a future palette/scale swap touches one file.

## Deviations from plan

### Auto-fixed (Rule 1 bugs surfaced while rendering)

**1. [Rule 1 — Bug] Figure.\_apply_jupyter_display_styling crash on Agg backend**
- **Found during:** Task 1 (first dispatcher run)
- **Issue:** `figure.canvas.layout` exists on every backend but is a *bound method* on Agg (`Figure.layout`), not the ipympl widget Layout. Calling `.width = ...` on a method raised `'builtin_function_or_method' object has no attribute 'width'`.
- **Fix:** Tightened the guard from `if hasattr(canvas, "layout")` to `if not hasattr(layout, "width")` — Agg's method gets skipped; ipympl's widget Layout passes through.
- **Files modified:** `research/dsplot/figure.py`
- **Commit:** 481d83b

**2. [Rule 1 — Bug] suptitle / panel-title overlap on multi-panel Figures**
- **Found during:** Task 1 (3-panel components_recombine)
- **Issue:** `Figure.render()` placed the suptitle at the default position and panel titles at axes y=1.02, so a 3-panel figure with a suptitle clipped both into the same band.
- **Fix:** Track `self._has_suptitle` in `Figure.__init__`; when set, `render()` calls `subplots_adjust(top=0.88)` to reserve the suptitle band.
- **Files modified:** `research/dsplot/figure.py`
- **Commit:** 481d83b

**3. [Rule 1 — Bug] StaticPanel title + subtitle stacked at the same y-position**
- **Found during:** Task 1 (panels carrying both title and subtitle — e.g. dot_product_geometry's per-panel angle case + result text)
- **Issue:** Title was painted with `ax.set_title(...)` at default y, then subtitle was placed at y=1.02 via `ax.text(...)`. They collided.
- **Fix:** When both are present, stack title at y=1.07 and subtitle at y=1.00 (≈ one line-height gap at DEFAULT_TITLE_FONT_SIZE). When only one is present, the original placement is preserved.
- **Files modified:** `research/dsplot/panels/static_panel.py`; `research/tests/dsplot/test_panel_composition.py` (asserts y=1.00 with explanatory comment)
- **Commit:** 481d83b (initial), 75a9867 (test assertion update)

**4. [Rule 1 — Bug] Annotation.draw raised TypeError on Axes3D**
- **Found during:** Task 2 (vector_projection_3d Path-1/Path-2 captions)
- **Issue:** `Axes3D.text(x, y, s)` requires three positional args (x, y, z); 2D `ax.text(x, y, s, transform=ax.transAxes)` raised "missing 1 required positional argument: 's'" when called on Axes3D.
- **Fix:** Branch on `hasattr(ax, "get_zlim")` — when 3D + `transform="axes"`, use `ax.text2D(x, y, s, transform=ax.transAxes, ...)`. 2D path is unchanged.
- **Files modified:** `research/dsplot/plottables/annotation.py`
- **Commit:** 75a9867

### Auto-fixed (Rule 3 blockers)

**5. [Rule 3 — Blocker] Missing reference audio file blocked compute_full_cwt**
- **Found during:** Task 3 (motivator render)
- **Issue:** `get_default_config()` validates `AudioConfig.file_path = "assets/audio/reference/prospa_murda_baby_sc_rip.wav"` exists; that file isn't in the worktree branch (gitignored / not propagated from main).
- **Fix:** Symlinked the file from the main repo working tree into the worktree at the same path. The symlink is NOT committed; it's transient runtime-only. The motivator/alignment_diagnostic modules don't actually USE the file — they just need the config to instantiate.
- **Files modified:** none (filesystem-only symlink, outside git)
- **Commit:** none

## Known stubs

**`components_recombine.py` Panel 3 subtitle = "PLACEHOLDER"** — the third panel of `components_recombine_either_order_v18.png` shows vector `a` alongside its mirrored sibling `a'` with perpendicular components. The figure exists but its subtitle copy is unwritten. Per project policy (user authors final prose; Claude scaffolds), the placeholder is intentional and the user will replace it during the merge-time copy pass.

## TDD gate compliance

Plan 09-05 is `type: implementation` (not `type: tdd`) — no RED/GREEN gate sequence applies. All commits are `feat` per the per-task atomic-commit policy. Library invariants are continuously enforced by the pre-existing 109-test dsplot suite, which passes after every task (`pytest research/tests/dsplot/ -q → 109 passed`).

## Self-Check: PASSED

Verified that every claimed output exists in the worktree:

| File | Exists |
| --- | --- |
| `research/dsplot/figures/motivator.py` | FOUND |
| `research/dsplot/figures/alignment_diagnostic.py` | FOUND |
| `research/dsplot/figures/vector_basics.py` | FOUND |
| `research/dsplot/figures/dot_product_geometry.py` | FOUND |
| `research/dsplot/figures/components_recombine.py` | FOUND |
| `research/dsplot/figures/projection_reconstruction.py` | FOUND |
| `research/dsplot/figures/vector_projection_3d.py` | FOUND |
| `research/dsplot/figures/foundation_constants.py` | FOUND |
| `research/dsplot/figures/__main__.py` | FOUND |
| `research/dsplot/panels/static_panel_3d.py` | FOUND |
| `assets/images/dsp/vector_basics_09_05.png` | FOUND |
| `assets/images/dsp/dot_product_geometry_09_05.png` | FOUND |
| `assets/images/dsp/components_recombine_either_order_v18_09_05.png` | FOUND |
| `assets/images/dsp/vector_xy_reconstruction_09_05.png` | FOUND |
| `assets/images/dsp/projection_reconstruction_either_order_v9_09_05.png` | FOUND |
| `assets/images/dsp/vector_projection_3d_v2_combo5_palette_09_05.png` | FOUND |
| `assets/images/generated/dsp_motivator_v4_100-2000hz_0.5s_09_05.png` | FOUND |
| `assets/images/generated/dsp_motivator_v5_50-5000hz_1.0s_09_05.png` | FOUND |
| `assets/images/generated/dsp_motivator_vw1_gentle_100-2000hz_0.5s_09_05.png` | FOUND |
| `assets/images/generated/dsp_motivator_vw2_moderate_100-2000hz_0.5s_09_05.png` | FOUND |
| `assets/images/generated/dsp_motivator_vw3_aggressive_100-2000hz_0.5s_09_05.png` | FOUND |
| `assets/images/generated/dsp_motivator_vw4_aggressive_20-20000hz_2.0s_09_05.png` | FOUND |
| `assets/images/generated/dsp_alignment_diagnostic_09_05.png` | FOUND |
| Commit 481d83b | FOUND in git log |
| Commit 75a9867 | FOUND in git log |
| Commit 25413db | FOUND in git log |

Test suite: `109 passed, 4 warnings in 0.77s`.

## Awaiting

This plan ends in a **visual parity checkpoint**. Every regenerated PNG ships to `assets/images/dsp/` or `assets/images/generated/` with the `_09_05` suffix so the originals coexist for side-by-side comparison.

The user verifies — at the post-return checkpoint — that each `_09_05.png` is visually equivalent to (or better than) the original at the same DSP.md filename. On approval, the merge-time rename strips `_09_05` from each filename and DSP.md continues to resolve unchanged.
