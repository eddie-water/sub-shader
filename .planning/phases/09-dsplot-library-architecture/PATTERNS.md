# Phase 9: dsplot Library Architecture — Pattern Map

**Mapped:** 2026-05-17
**Scope:** Inventory every plotting primitive in `research/` (and any matplotlib code outside `src/`) and map it onto the planned `dsplot` two-layer architecture (Plottables + Panels).

## Proposed Target Layout

| Module | Role |
|---|---|
| `dsplot/style.py` | Generic style constants (colors, fontsizes, linewidths, sizing). Single source of truth. Reassignable. |
| `dsplot/primitives.py` | Low-level draw helpers shared by multiple Plottables (axes scaffolds, sharp-arrow patch, dashed dropline, label placement, log-freq tick mapper). |
| `dsplot/plottables/` | One file per Plottable class: `vector.py`, `vector_components.py`, `vector_3d.py`, `time_series.py`, `heatmap.py`, `spotlight.py`, `dropline.py`, `annotation.py`. |
| `dsplot/panels/` | `static.py`, `dynamic.py` (FuncAnimation), `interactive.py` (ipywidgets), plus a shared `base.py`. |
| `dsplot/figures/` | Canned compositions for the four foundation figures + motivator + alignment diagnostic. Each is a thin `compose(...)` function returning a `Figure`. |
| `research/utilities/` (kept) | Non-plotting utilities only: timing, printing, dsp_helpers (signal synthesis), signals registry, wav_export, constants (paths). |
| RETIRED | Dead helpers + the host-coupled `compute_full_cwt` (stays in research/utilities but is NOT pulled into dsplot — dsplot has zero subshader imports). |

---

## File Classification

| Source file | Role | Verdict |
|---|---|---|
| `research/dsp_figures.py` | Monolith: 5 dispatched figures + 4 retired figures + 3 shared private helpers + module-level constants | DECOMPOSE into `dsplot/figures/*` + `dsplot/primitives.py` + Plottables |
| `research/utilities/plotting.py` | Mixed bag: spectrogram helpers, atomic single-axes plotters, scaffold builders, vector helpers, CWT wrapper | SPLIT across `dsplot/primitives.py`, `dsplot/plottables/*`, and (CWT wrapper) STAYS in research/utilities |
| `research/utilities/style.py` | Style constants — already centralized but uses subshader-flavored names | RENAME → `dsplot/style.py` (genericize names per "Style constants to genericize" below) |
| `research/utilities/constants.py` | File paths, audio defaults, GPU detection | STAYS in research/utilities (not plotting) |
| `research/utilities/dsp_helpers.py` | Chirp signal synthesis (bouncing, waypoint, FM, wandering, linear) | STAYS in research/utilities (not plotting) |
| `research/utilities/printing.py` | Console table/progress printing | STAYS in research/utilities (not plotting) |
| `research/utilities/signals.py` | Audio signal registry | STAYS in research/utilities (not plotting) |
| `research/utilities/timing.py` | `time_call`, `TimingAccumulator` | STAYS in research/utilities (not plotting) |
| `research/utilities/wav_export.py` | WAV writer | STAYS in research/utilities (not plotting) |
| `research/utilities/__init__.py` | Re-export hub | KEEP, prune plotting re-exports once plotting.py is moved |
| `research/comparison.py` | 5×3 comparison grid + timing bar chart | OUT OF SCOPE for Phase 9 (imports subshader directly); however, its `Heatmap`/`TimeSeries` usage patterns inform Plottable API |
| `research/figures.py` | `ReadmeFigures` class — chirp/polyphonic/musical comparison PNGs | OUT OF SCOPE for Phase 9 (imports subshader directly); same pattern-source role as comparison.py |
| `research/_temp_dot_product_symmetry_candidates.py` | Throwaway scratch script | RETIRE (delete or leave; never port) |
| `research/palette_schema.py` | Palette design tool with swatches/labels | STAYS in research/ as standalone palette tooling (not a Plottable; uses matplotlib but is itself a one-off visualizer) |
| `research/test_suite.py` | CLI entry point that imports matplotlib inside conditionals | STAYS; will pick up new `dsplot.figures` callsite for `--dsp-figures` flag |
| `src/subshader/utils/signal_generator.py` | matplotlib import inside `src/` | OUT OF SCOPE (Phase 9 cannot touch src/) |
| `src/subshader/dsp/wavelet_kernel.py` | matplotlib import inside `src/` | OUT OF SCOPE (Phase 9 cannot touch src/) |

---

## Pattern Assignments — dsp_figures.py

### Module-level constants (`dsp_figures.py:421-432`)
- `A`, `A_PRIME`, `B`, `A_Z`, `FOUND_LIM` — figure-defining vector values
- `_DIM_NEUTRAL`, `_DIM_SPINE`, `_DIM_X_COLOR`, `_DIM_Y_COLOR`, `_DIM_Z_COLOR` — local aliases on top of `style.PALETTE_*`
- **Migration:** keep with the figure compositions (e.g., `dsplot/figures/foundations.py`) since they are figure-specific, not library style. Per CONTEXT they are NOT library-level style constants.
- **Cross-cut:** any value change must be mirrored into `src/subshader/dsp/DSP.md` alt-text + LaTeX block. Flag this in the migration task.

### `_save(fig, output_dir, filename)` (`dsp_figures.py:435-441`)
- Generic save wrapper: facecolor, dpi, bbox_inches, pad.
- **New home:** `dsplot/panels/base.py` as `Panel.save(path, **kwargs)` method OR `dsplot/figures/_io.py` helper. Used by every figure.

### `_plot_sharp_vector(ax, vec, ...)` (`dsp_figures.py:505-616`)
- `FancyArrowPatch`-based arrow with sharp head, optional dashed shaft, flexible label offset (scalar OR (dx,dy) OR `label_dx`/`label_dy` nudges).
- **New home:** `dsplot/primitives.py` as `draw_sharp_arrow(ax, ...)`. Becomes the default draw primitive used by `Vector.draw()` (the project's preferred arrow style — the older `plot_vector` in plotting.py uses `ax.annotate` and has less control over head size).
- **Cross-cut:** also used by `VectorComponents` and `Dropline` rendering; lives in primitives.

### `_panel_titles(ax, title, subtitle)` (`dsp_figures.py:619-637`)
- Two-tier panel title (title + italic subtitle below) using canonical font hierarchy.
- **New home:** method on `Panel` base class (e.g., `panel.set_titles(title, subtitle=None)`), OR primitive `apply_panel_titles(ax, title, subtitle)` in `dsplot/primitives.py`. Lean toward Panel method since titling is a Panel-level concern.

### `_draw_dashed_tip_to_tail(ax, vec, first_axis="x", ...)` (`dsp_figures.py:640-661`)
- Draws the two-segment x-then-y or y-then-x reconstruction of a vector as dashed neutral component arrows.
- **New home:** becomes the `draw()` logic of `VectorComponents` Plottable. Constructor takes the source vector + `first_axis`; `draw()` calls `draw_sharp_arrow` twice.

### `_plot_vector_xy_projection` (`dsp_figures.py:444-502`)
- RETIRED in dispatch. Hardcoded vector, not tied to canonical A.
- **Verdict:** RETIRE. Pedagogy is now covered by `_plot_vector_xy_reconstruction` panel 1.

### `_plot_vector_xy_reconstruction` (`dsp_figures.py:664-787`)
- 3-panel "Basic Vector Projection" (figure 1). Uses A, A_PRIME, FOUND_LIM. Suptitle + per-panel two-tier titles. Calls `_draw_dashed_tip_to_tail`, `_plot_sharp_vector`, droplines, labels.
- **New home:** `dsplot/figures/components_recombine.py`. Composition:
  - `StaticPanel` × 3 in a row
  - Each panel: `VectorAxes` setup (`primitives.setup_vector_axes`), `Dropline` × 2 (panel 1 only), `VectorComponents(A, first_axis="x")`, `VectorComponents(A, first_axis="y")` for panel 2 both, `Vector(A, color=PRIMARY)`, `Vector(A_PRIME, color=PRIMARY)` for panel 3, plus `Annotation` labels for each component edge.
- **Primary candidate for the `DynamicPanel` notebook demo** (CONTEXT verification target 3): vary A across frames while A_PRIME's y-component stays fixed.

### `_plot_vector_basics` (`dsp_figures.py:790-819`)
- Single panel, 5 arrows of varying magnitude/direction. No labels, no math.
- **New home:** `dsplot/figures/vector_basics.py`. Composition: single `StaticPanel`; loop adds 5 `Vector(sample, color=...)` Plottables.

### `_plot_dot_product_geometry` (`dsp_figures.py:822-876`)
- 4-panel angle-cases figure. Each panel: 2 vectors + panel title + result-text annotation.
- **New home:** `dsplot/figures/dot_product_geometry.py`. Composition: 4 `StaticPanel`s in a row; each adds 2 `Vector` Plottables; panel title via `panel.set_titles`; result annotation via `Annotation` (bottom-centered axes-coords text).

### `_plot_vector_similarity` (`dsp_figures.py:879-911`)
- RETIRED. Two oblique pairs. Absorbed by the oblique panel of `_plot_dot_product_geometry`.
- **Verdict:** RETIRE.

### `_plot_vector_projection` (`dsp_figures.py:914-949`)
- RETIRED. Long/short shadow demo. Message carried by 4-panel sign-cases.
- **Verdict:** RETIRE.

### `_plot_projection_reference_directions` (`dsp_figures.py:952-1106`)
- RETIRED. 3-panel juxtaposition (x/y axes + a-onto-b + b-onto-a). Split into two newer figures.
- **Verdict:** RETIRE. Its inner `_draw_proj` nested helper is superseded by `_draw_reconstruction` in `_plot_projection_reconstruction_either_order`.

### `_plot_dot_product_symmetry` (`dsp_figures.py:1109-1196`)
- RETIRED. 2-panel projection-only. Superseded by `_plot_projection_reconstruction_either_order`.
- **Verdict:** RETIRE.

### `_plot_projection_reconstruction_either_order` (`dsp_figures.py:1199-1336`)
- 2-panel "Vector Projection Symmetry" (figure 2). Uses A, B, FOUND_LIM. Calls `_plot_sharp_vector`, `_panel_titles`, custom `_draw_reconstruction` nested helper that decomposes one vector onto another into parallel + perpendicular components and draws two dashed reconstruction paths.
- **New home:** `dsplot/figures/projection_reconstruction.py`. Composition: 2 `StaticPanel`s; each composes:
  - Reference `Vector(target, color=...)`
  - Two parallel/perp `Vector(component, color=NEUTRAL)` (component arrows as shadow)
  - Two `Dropline`s closing the parallelogram (par-tip → src-tip and per-tip → src-tip)
  - Subject `Vector(source, color=...)` on top
- The `_draw_reconstruction` nested helper becomes a private composition method `_compose_projection_reconstruction(panel, source, target, ...)` in the figure module (NOT a Plottable — it orchestrates 5+ Plottables into a parallelogram).

### `_plot_vector_projection_3d` (`dsp_figures.py:1339-1506`)
- 3D figure: manual spines through origin, two reconstruction paths (x→y→z and z→y→x) with per-segment role color, vector A as neutral solid line + scatter dot, 2D order legend, hidden mpl axis chrome.
- **New home:** introduces a NEW Plottable not in the CONTEXT list: `Vector3D`. CONTEXT names 7 Plottables but the 3D figure is explicitly in the first-delivery scope, so a `Vector3D` Plottable belongs in `dsplot/plottables/vector_3d.py`. Also requires:
  - `dsplot/primitives.py::setup_3d_axes(ax, lim, ...)` — manual spines + labels + axis_off + view_init
  - `Vector3D` Plottable that draws a single 3D arrow as `ax.plot` + `ax.scatter` (mpl 3D `quiver` under-renders on dark backgrounds — keep the project's workaround)
  - `VectorComponents3D` Plottable for the dashed reconstruction paths (extension of 2D `VectorComponents`)
- **Risk:** the 3D Plottable + the path-color-per-segment pedagogy is more complex than 2D Vector. See Migration Risks below.

### `generate_motivator_versions` / `render_motivator` (`dsp_figures.py:165-327`)
- 6 versions of a 3-row motivator figure (TimeSeries+inst-freq twin / STFT / CWT). Uses `MOTIVATOR_VERSIONS` registry. Calls `compute_full_cwt` (which has subshader imports — host-only).
- **New home:** `dsplot/figures/motivator.py` for the COMPOSITION (`TimeSeries` Plottable on row 0, `TimeSeries` again on the twin for inst-freq, `Heatmap` on rows 1-2). The CWT/STFT computation stays in `research/utilities/plotting.py::compute_full_cwt`; the figure composer accepts already-computed arrays and renders them.
- `MOTIVATOR_VERSIONS` list of `ChirpFigureConfig` dataclasses stays in `dsplot/figures/motivator.py` since they're figure-defining values, not style.
- **Pattern split:** "compose figure" lives in dsplot; "compute the data" stays in research/utilities (subshader-dependent).

### `generate_alignment_diagnostic` (`dsp_figures.py:334-394`)
- 2-row figure: time series + CWT heatmap with cyan vertical line overlay marking burst time + dotted horizontals at each burst freq.
- **New home:** `dsplot/figures/alignment_diagnostic.py`. Composition: 2 `StaticPanel`s vertically; row 0 = `TimeSeries` + `Spotlight` (vertical-line variant); row 1 = `Heatmap` + `Spotlight` (vertical line) + three `Spotlight` (horizontal lines for burst freqs) + a `legend`. This is the **canonical Spotlight use case** — cyan overlay is exactly the "highlight overlay" the Spotlight Plottable was specced for.
- Data computation (`compute_full_cwt` call) stays out of dsplot.

### `generate_foundations_figures` (`dsp_figures.py:1509-1542`)
- Dispatch table: calls 5 retained foundation figures.
- **New home:** `dsplot/figures/__init__.py::generate_foundations_figures()` thin dispatcher, OR retire entirely if the per-figure `compose()` functions are called directly from `research/dsp_figures.py` (which becomes a deprecation shim per CONTEXT verification target 6).

### `generate_all_dsp_figures` (`dsp_figures.py:401-407`)
- Top-level dispatch: motivators + alignment + foundations.
- **New home:** Stays as a top-level helper in `dsplot/figures/__init__.py` OR in the post-refactor `research/dsp_figures.py` shim.

---

## Pattern Assignments — utilities/plotting.py

### `compute_freq_yticks(cwt_freqs, tick_freqs)` (`plotting.py:25-37`)
- Maps log-spaced frequencies to bin positions for spectrogram y-axis labeling.
- **New home:** `dsplot/primitives.py` as `log_freq_tick_positions(freqs, ticks)`. Shared by `Heatmap` and `TimeSeries` (inst-freq overlay).

### `placeholder_ax(ax, label)` (`plotting.py:40-49`)
- Renders a "[ reference not available ]" placeholder with diagonal hatch.
- **New home:** `dsplot/primitives.py::draw_placeholder(ax, label)`. Or absorbed into a `PlaceholderImage` Plottable variant if image-missing cases need first-class support. Lean toward primitive — used inline by panels.

### `create_figure_scaffold(title, subtitle, n_top_rows)` (`plotting.py:52-74`)
- Builds a stacked-rows-with-shared-spectrogram figure scaffold for the README comparison figures.
- **New home:** **OUT OF SCOPE for first delivery** — used by `figures.py` (which is subshader-coupled and not in Phase 9). Leave in research/utilities for now. Can be ported to a `MultiPanelFigure` builder in a future phase if a non-subshader caller needs it.

### `render_top_row(fig, gs, idx, row, ax_stft, ...)` (`plotting.py:77-120`)
- Dispatches waveform / freq_line / image row rendering into a shared gridspec.
- **New home:** OUT OF SCOPE for first delivery (only used by figures.py / comparison.py). Stays in research/utilities. The conditional dispatch by `row["type"]` is exactly the kind of imperative branching that the Plottable architecture replaces — callers will instead compose `TimeSeries` / `Heatmap` / `ImageReference` Plottables on a Panel directly.

### `render_spectrogram_row(ax, data, ...)` (`plotting.py:123-137`)
- Imshow + log-freq ticks + bottom-row-aware x label.
- **New home:** core logic becomes `Heatmap.draw()` in `dsplot/plottables/heatmap.py`. The "is_bottom" flag becomes Panel-level (Panel decides whether to show xtick labels based on position in a multi-panel figure).

### `downsample_spec(arr, max_rows, max_cols)` (`plotting.py:140-154`)
- Downsamples a 2D array via scipy_resample for heatmap rendering.
- **New home:** `dsplot/primitives.py::downsample_2d(arr, max_rows, max_cols)` OR an internal helper of `Heatmap`. Lean toward Heatmap-internal (only consumer).

### `plot_time_series(ax, signal, sr, ...)` (`plotting.py:161-171`)
- Fills audio waveform vs time on a single axes with peak-padding y-lim, dark bg.
- **New home:** core of `TimeSeries.draw()` in `dsplot/plottables/time_series.py`.

### `plot_inst_freq(ax, inst_freq_hz, t, cwt_freqs, ...)` (`plotting.py:174-189`)
- Plots instantaneous-frequency curve in bin-space with log-freq y ticks + grid.
- **New home:** also `TimeSeries` Plottable, with a `y_mode="bin_space"` or similar variant; OR a dedicated `FrequencyCurve` Plottable subclass of `TimeSeries`. Recommendation: single `TimeSeries` with a `y_mapping=` constructor kwarg (default linear; bin-space when `cwt_freqs` provided).

### `plot_fft_magnitude(ax, signal, sr, ...)` (`plotting.py:192-208`)
- One-sided FFT magnitude with log-log default.
- **New home:** OUT OF SCOPE for first delivery (no current Phase-9 figure uses it). Could become `FFTSpectrum` Plottable in a follow-up phase. Leave in research/utilities.

### `plot_stft_spectrogram(ax, signal, sr, ...)` (`plotting.py:211-260`)
- STFT magnitude on log-freq y with `pcolormesh`, vmax percentile, freq-limit clamping.
- **New home:** the data computation (`scipy_stft` + DC bin drop + vmax pick) is NOT plotting — should live in `research/utilities/dsp_helpers.py` as `compute_stft_magnitude(signal, sr, ...) -> (f, t, mag)`. The visual render then becomes `Heatmap` with `y_scale="log"` + explicit `freq_lim` + log-freq tick mapping. **Split compute from render.**

### `plot_cwt_spectrogram(ax, cwt_data, duration_s, cwt_freqs, ...)` (`plotting.py:263-277`)
- Imshow of pre-computed CWT data with log-freq tick mapping.
- **New home:** `Heatmap.draw()`. No splitting needed — already takes pre-computed data.

### `render_image_row(ax, img_path, fallback_label, extent)` (`plotting.py:280-293`)
- Loads a PNG into an axes with placeholder fallback.
- **New home:** `dsplot/plottables/image_reference.py::ImageReference` (or stretch the `Annotation` Plottable to accept image paths — but image-as-Plottable is cleaner). NOT in CONTEXT's 7-Plottable list; introduce only if foundations or motivator figures actually need it. Current dispatched figures DO NOT use image rows; SKIP for first delivery and leave in research/utilities for `figures.py`/`comparison.py`.

### `create_grid_scaffold(n_rows, n_cols, ...)` (`plotting.py:301-338`)
- N×M grid of axes with dark bg and gray spines, supports width/height ratios.
- **New home:** `dsplot/panels/grid.py::PanelGrid(n_rows, n_cols, ...)` — a layout helper that yields per-cell Panels. Used by motivator (3×2 with hidden label column) and alignment diagnostic (2×1).
- **Cross-cut:** Panel composition (any panel type can be placed in any grid cell — supports CONTEXT's "mixed-type figures" goal).

### `create_panel_row(n_panels, panel_size, height, suptitle, ...)` (`plotting.py:348-376`)
- 1×N row of square axes for vector/polar/small-figure layouts.
- **New home:** `dsplot/panels/grid.py::PanelRow(n_panels, ...)` — convenience wrapper around `PanelGrid(1, n_panels)` with vector-figure default sizing.
- **Cross-cut:** suptitle handling becomes a `Figure`-level concern, not a panel-level one. The Figure object (which holds Panels) owns the suptitle.

### `setup_vector_axes(ax, lim, panel_title, result_text, ...)` (`plotting.py:384-465`)
- Square axes with origin crosshair (or arrow-style axes), optional x/y labels, optional panel title + result annotation. Heavy use of style.VECTOR_* constants.
- **New home:** SPLIT:
  - Pure axes setup (limits, aspect, ticks-off, spines, axhline/axvline OR arrow-style axes) → `dsplot/primitives.py::setup_vector_axes(ax, lim, ...)` or a method on `VectorPanel(StaticPanel)`.
  - Panel title + subtitle → `Panel.set_titles()` method (replaces `_panel_titles` from dsp_figures).
  - Result text annotation → `Annotation` Plottable composed by caller.
- **Recommendation:** create a thin `VectorPanel` subclass of `StaticPanel` that auto-applies vector-axis setup on construction (`lim=`, `axis_style="arrow"|"line"`, `axis_labels=`), so the figure code reads `panel = VectorPanel(lim=FOUND_LIM, axis_style="arrow", axis_labels=True)` instead of `panel = StaticPanel(); setup_vector_axes(panel.ax, ...)`.

### `plot_vector(ax, vec, origin, color, label, ...)` (`plotting.py:468-511`)
- `ax.annotate`-based arrow with `arrowstyle="-|>"`, optional label past tip.
- **New home:** core of `Vector.draw()` in `dsplot/plottables/vector.py`. **However,** `_plot_sharp_vector` from dsp_figures.py is the newer, more capable arrow primitive (FancyArrowPatch, sharp head, dashed-shaft solid-head split, flexible label). Recommend the Plottable internally uses the FancyArrowPatch path as its default; the older annotate-based logic can be retired.
- **Cross-cut:** used by `VectorComponents` (which draws 2 vectors per call) and the projection-reconstruction figure's `_draw_reconstruction` composer (draws ~5 Vectors per panel).

### `plot_projection(ax, a, b, ...)` (`plotting.py:514-552`)
- Draws a, b, and the scalar projection of b onto a, plus optional dropline.
- **New home:** Becomes a composition pattern, NOT a single Plottable. Three Plottables composed:
  - `Vector(a, color=A_COLOR)`
  - `Vector(b, color=B_COLOR)`
  - `Vector(foot, color=PROJ_COLOR)` (the shadow)
  - `Dropline(from=b_tip, to=foot_pos)` (perpendicular dropline)
- Lives as `dsplot/figures/_projection_helpers.py::compose_projection(panel, a, b)` (helper for figure composers, not a Plottable).
- Currently used only by the RETIRED `_plot_vector_projection` figure — verify no live caller before pulling forward.

### `compute_full_cwt(signal, sr, ...)` (`plotting.py:559-649`)
- **CRITICAL: imports `from subshader.config`, `from subshader.dsp.cwt`, `from subshader.renderer.frame_buffer`.**
- Drives the full audio → CWT pipeline including chunking + buffer stitching.
- **New home:** STAYS in `research/utilities/plotting.py` (or move to `research/utilities/dsp_helpers.py` for cleaner home). MUST NOT enter `dsplot/` — the CONTEXT Non-Goal #1 forbids subshader imports anywhere in dsplot. Figure compositions in `dsplot/figures/motivator.py` and `alignment_diagnostic.py` accept pre-computed CWT arrays as input; the *caller* (research/dsp_figures.py shim) bridges from `compute_full_cwt` to the dsplot figure composer.

---

## Pattern Assignments — utilities/style.py

### Existing structure (lines 1-138)
- COLORS section
- FONT SIZES
- LINE WIDTHS
- FIGURE DIMENSIONS
- LAYOUT SPACING
- COMPARISON GRID
- RENDERING (DPI)
- ATOMIC PLOTTERS (tick colors, spines, grid, percentiles, padding)
- DSP.md FOUNDATION FIGURES — VECTOR PLOTS (palette + per-figure vector knobs)
- DSP.md MOTIVATOR FIGURE (compact 3-row knobs)

### Migration
- **Whole file → `dsplot/style.py`**, but with renaming (see below) and removal of subshader-specific names.
- Comparison-grid + motivator + foundation-figure subsections can either stay grouped in `style.py` or split into `dsplot/style/colors.py + typography.py + sizing.py` (Claude's discretion per CONTEXT). **Recommendation:** single `style.py` for first delivery — easier to override via "import + reassign constants" (the CONTEXT-locked override pattern).
- Constants like `MOTIVATOR_VISIBLE_END_S` and `LABEL_CHAR_WIDTH` are figure-specific tuning — keep in style.py but document them as such, OR push into the relevant `dsplot/figures/<theme>.py` module.

---

## Shared Patterns

### Pattern: "Figure-defining constants live with the figure, library style lives in style.py"
- Applies to: A, A_PRIME, B, A_Z, FOUND_LIM (foundations); MOTIVATOR_VERSIONS list (motivator); CHIRP_F0/F1 (comparison — but that's out of scope).
- **Rule:** if changing the value changes the visual content of one figure → it's figure-local. If changing it changes the look of *every* figure → it's style.

### Pattern: "Compute vs render split"
- Applies to: `plot_stft_spectrogram` (computes STFT then renders), `compute_full_cwt` (the heavy hitter), any future CWT/FFT figure.
- **Rule:** dsplot accepts pre-computed arrays. Computation that needs subshader (or scipy, or cupy) lives outside dsplot in `research/utilities/` or in the caller's notebook cell. dsplot only knows about numpy arrays and matplotlib axes.

### Pattern: "Panel-level concerns vs Plottable-level concerns"
- **Panel-level:** axes setup (limits, aspect, ticks visibility, spine colors, bg color), titles + subtitles, save logic, animation/widget orchestration, layout in a grid.
- **Plottable-level:** what to draw (arrow, line, image, scatter, text), style (color, linewidth, alpha, linestyle, z-order), data it represents.
- **Cross-cut:** the `setup_vector_axes` function currently does BOTH — split per the assignment above.

### Pattern: "Composition helpers for multi-Plottable patterns"
- Applies to: `_draw_reconstruction` (5 Vectors + 2 Droplines per panel), `compose_projection` (3 Vectors + 1 Dropline), `_draw_dashed_tip_to_tail` (2 dashed Vector arrows — though this one is simple enough to be its own Plottable, `VectorComponents`).
- **Rule:** if the composition is **reusable** (used across multiple figures) and **conceptually one unit** (e.g., "the components of vector A"), make it a Plottable. If it's **figure-specific** (e.g., "the parallelogram for projecting source onto target"), keep it as a private composer function in the figure module.

### Pattern: "Per-axis tweaks for shared visual style"
- Applies to: spine colors/widths, tick label colors, axis grid colors. Currently set inline in dsp_figures.py twin-axis code AND in comparison.py's spine loops.
- **Rule:** Panel base class applies these on construction via `style.SPINE_COLOR` / `style.SPINE_LINEWIDTH` etc. Plottables that want to override push into their own `draw()` (e.g., `Vector` doesn't touch spines, but `Heatmap` might want to set a different bg).

---

## Style constants to genericize

CONTEXT Non-Goal #5: "Style constants are generic (e.g. `PRIMARY_COLOR`, not `WAVELET_ORANGE`)." Current style.py has SubShader-flavored names. Proposed renames:

| Current name | Generic name | Notes |
|---|---|---|
| `WAVEFORM_COLOR` | `TIMESERIES_COLOR` | "Waveform" is audio-specific; TimeSeries is the Plottable name |
| `WAVEFORM_ALPHA` | `TIMESERIES_ALPHA` | same |
| `WAVEFORM_YLIM_PADDING` | `TIMESERIES_YLIM_PADDING` | same |
| `GRID_WAVEFORM_COLOR` | `OVERLAY_LINE_COLOR` | Currently used for inst-freq overlay; "grid waveform" is opaque |
| `FREQ_LINE_COLOR` | `OVERLAY_LINE_COLOR_ALT` | Currently for freq curves on spectrograms; consolidate w/ above? |
| `FREQ_LINE_WIDTH` | `OVERLAY_LINEWIDTH` | same |
| `INST_FREQ_LINEWIDTH` | `OVERLAY_LINEWIDTH_THICK` | merge with above if possible |
| `INST_FREQ_ALPHA` | `OVERLAY_ALPHA` | same |
| `FFT_LINEWIDTH` | `LINE_LINEWIDTH` | generic line weight |
| `GRID_CMAP` | `HEATMAP_CMAP` | "Grid" implies comparison-grid; cmap is heatmap concern |
| `GRID_*` (HSPACE, WSPACE, MARGIN, FIGSIZE_W, FIGSIZE_H, TITLE_PAD) | `PANELGRID_*` | "Grid" overloaded; PanelGrid is the layout class |
| `LABEL_CHAR_WIDTH`, `LABEL_PAD` | `ROW_LABEL_CHAR_WIDTH`, `ROW_LABEL_PAD` | clarify they're for row-label column sizing |
| `VECTOR_A_COLOR` | `PRIMARY_COLOR` (alias to PALETTE_PRIMARY) | already aliased; promote alias to canonical name |
| `VECTOR_B_COLOR` | `SECONDARY_COLOR` (alias to PALETTE_SECONDARY) | same |
| `VECTOR_PROJ_COLOR` | (RETIRE) | "Legacy — projection / shadow." Comments say new figures use neutral; not needed |
| `VECTOR_NEUTRAL_COLOR` | `NEUTRAL_COLOR` | drop the VECTOR_ prefix; it's used by axes/shadows too |
| `VECTOR_AXIS_COLOR` | `SPINE_COLOR_VECTOR` or merge into existing `SPINE_COLOR` | check if a separate vector-axes shade is actually needed |
| `VECTOR_AXIS_ALPHA` | drop or merge with `AXIS_GRID_ALPHA` | similar |
| `VECTOR_DROPLINE_COLOR`, `VECTOR_DROPLINE_ALPHA` | `DROPLINE_COLOR`, `DROPLINE_ALPHA` | Dropline is a generic Plottable |
| `VECTOR_LINEWIDTH`, `VECTOR_BOLD_LINEWIDTH` | `ARROW_LINEWIDTH`, `ARROW_LINEWIDTH_BOLD` | Vector → Arrow (Vector is the Plottable; arrow is the visual atom) |
| `VECTOR_HEAD_WIDTH`, `VECTOR_HEAD_LENGTH` | `ARROW_HEAD_WIDTH`, `ARROW_HEAD_LENGTH` | same |
| `VECTOR_LABEL_FONT_SIZE`, `VECTOR_LABEL_OFFSET` | `LABEL_FONT_SIZE`, `LABEL_OFFSET` | generic |
| `VECTOR_PANEL_TITLE_SIZE` | `PANEL_TITLE_SIZE` | generic |
| `VECTOR_PANEL_RESULT_SIZE` | `PANEL_ANNOTATION_SIZE` | "result" is foundations-specific; "annotation" generic |
| `VECTOR_PANEL_RESULT_COLOR` | `PANEL_ANNOTATION_COLOR` | same |
| `VECTOR_DEFAULT_LIM` | (move to figures/foundations.py) | This is a figure-level default, not library style |
| `VECTOR_FIGSIZE_PER_PANEL`, `VECTOR_FIGSIZE_HEIGHT` | `SQUARE_PANEL_SIZE`, `SQUARE_PANEL_HEIGHT` | generic square-panel sizing |
| `VECTOR_ORANGE`, `VECTOR_BLUE` | (RETIRE) | "Matplotlib tableau palette" comment says these are for hand-tuning; not used by dispatched figures. Verify and retire. |
| `VECTOR_AXIS_ARROW_INSET`, `VECTOR_AXIS_LABEL_OFFSET`, `VECTOR_AXIS_LABEL_SIZE` | `AXES_ARROW_INSET`, `AXES_LABEL_OFFSET`, `AXES_LABEL_SIZE` | drop VECTOR_ prefix |
| `MOTIVATOR_*` (5 constants) | (move to figures/motivator.py) | Figure-specific tuning, not library style |
| `PALETTE_PRIMARY/SECONDARY/TERTIARY` | KEEP | Already generic — these become the canonical role-color names |
| `BG_COLOR`, `SUBTITLE_COLOR`, `TITLE_FONT_SIZE`, `TICK_LABEL_SIZE`, `AXIS_LABEL_FONT_SIZE`, `SUPTITLE_FONT_SIZE`, `SUBTITLE_FONT_SIZE`, `LABEL_FONT_SIZE` | KEEP | Already generic |
| `HSPACE`, `LEFT_MARGIN`, `RIGHT_MARGIN`, `BOTTOM_MARGIN`, `TOP_MARGIN`, `SUPTITLE_Y`, `SUBTITLE_Y` | KEEP | Generic layout |
| `DEFAULT_DPI`, `STUB_DPI` | KEEP | Generic |
| `TICK_LABEL_COLOR`, `SPINE_COLOR`, `SPINE_LINEWIDTH`, `AXIS_GRID_COLOR`, `AXIS_GRID_ALPHA`, `AXIS_GRID_LINEWIDTH` | KEEP | Generic |
| `VMAX_PERCENTILE` | KEEP | Generic image-rendering knob |
| `FIGURE_WIDTH`, `ROW_HEIGHT` | (move to research/utilities or rename) | These are for the multi-row comparison figure layout — figure-specific |

**Critical compatibility note:** the renames break `research/comparison.py`, `research/figures.py`, and any other consumer. CONTEXT verification target 6 says "Old module retired: research/dsp_figures.py is either deleted or reduced to a deprecation shim." Apply same approach: keep `research/utilities/style.py` as a shim that re-exports from `dsplot.style` with backward-compat aliases for `WAVEFORM_COLOR` → `TIMESERIES_COLOR` etc., so comparison.py and figures.py keep working.

---

## Migration Risks

### Risk 1: `compute_full_cwt` is the bridge between subshader and figures
- **Problem:** Every dispatched figure that needs CWT data calls `compute_full_cwt`, which imports `subshader.config`, `subshader.dsp.cwt`, `subshader.renderer.frame_buffer`. CONTEXT forbids these imports in dsplot.
- **Mitigation:** dsplot figure composers take pre-computed arrays. The bridge (calling `compute_full_cwt` then handing arrays to dsplot) lives in `research/dsp_figures.py` (the shim) or in the notebook cell. Confirm this two-step pattern with user before locking the API.

### Risk 2: `_plot_sharp_vector` has subtle dashed-shaft + solid-head split logic
- **Problem:** When linestyle != "-", the function splits the arrow into a dashed shaft (ax.plot) + a tiny solid FancyArrowPatch stub for the head. Otherwise the dash pattern would apply to the arrowhead outline and ghost the head. Visual fidelity depends on the 1e-3 stub length.
- **Mitigation:** Port the split logic intact into `Vector.draw()` when `linestyle != "-"`. Include a visual regression test (CONTEXT verification target 2 — image-diff against current dashed component arrows in panel 2 of figure 1).

### Risk 3: `_plot_vector_projection_3d` introduces an 8th Plottable (Vector3D) beyond CONTEXT's 7
- **Problem:** CONTEXT specifies 7 Plottables; 3D figure needs `Vector3D` + `VectorComponents3D`. Adding new classes risks violating Non-Goal #2 ("No new plot types beyond the 7 Plottables").
- **Mitigation:** Flag for user. Options: (a) extend the existing `Vector` and `VectorComponents` Plottables to handle both 2D and 3D internally via constructor arity (3-tuple → 3D), OR (b) introduce `Vector3D` as a separate Plottable and update CONTEXT's count. Option (a) is cleaner architecturally; option (b) is honest about the dimensionality difference. **Recommend (a)** — same Plottable, polymorphic on input dimension.

### Risk 4: Custom 3D spine/label/view-init logic is tightly coupled
- **Problem:** `_plot_vector_projection_3d` hides default mpl 3D chrome via `set_axis_off()`, draws 3 manual spines, sets `view_init(elev=38, azim=-55)` to avoid mpl 3.10's positive-azim projection collapse, and uses `ax.plot + ax.scatter` instead of `ax.quiver` because thin lines under-render. All of this needs to land somewhere — likely a `Vector3DPanel(StaticPanel)` class.
- **Mitigation:** Encapsulate in a dedicated `Vector3DPanel` subclass. Document the matplotlib version quirks in the class docstring so they don't get "cleaned up" by mistake later.

### Risk 5: `MOTIVATOR_VERSIONS` list has 6 ChirpFigureConfig entries with two different config dataclass types (Bouncing + Waypoint)
- **Problem:** The dispatcher branches on `isinstance(cfg.chirp, WaypointChirpConfig)` to call the right `build_*_chirp` function. Moving the *figure composition* to dsplot leaves this branch in the caller — but where? In `research/dsp_figures.py` shim? In a notebook helper?
- **Mitigation:** Two layers — `dsplot/figures/motivator.py::compose_motivator(signal, inst_freq, t, cwt_data, cwt_freqs, sr, output_filename)` takes pre-computed everything; `research/dsp_figures.py` shim retains the `render_motivator(cfg)` dispatch that picks the chirp builder, calls `compute_full_cwt`, and hands the arrays to `compose_motivator`. Net: dsplot has no chirp-config knowledge.

### Risk 6: Visual-fidelity regression across 6 figures
- **Problem:** CONTEXT verification target 2 is "Every figure currently produced by `research/dsp_figures.py` is reproducible from `dsplot` with identical or better visual fidelity." 6 dispatched outputs + visual diff is the gate. Subtle differences (e.g., arrow head size, label offsets, dashed shaft pattern) can sneak in if Plottable defaults don't exactly mirror current call-site values.
- **Mitigation:** During port, generate before/after PNGs for each figure. Use a side-by-side image diff (manual or perceptual hash) per CONTEXT Claude's-discretion item. Lock visual constants once parity is achieved.

### Risk 7: `setup_vector_axes` has many call-site overrides
- **Problem:** Different figures pass different combinations of `lim`, `panel_title`, `result_text`, `show_border`, `axis_style`, `axis_labels`, `x_color`, `y_color`, `axis_alpha`, `axis_linewidth`. Splitting into Panel-method + Annotation-Plottable risks dropping one of these knobs.
- **Mitigation:** Enumerate every existing call-site (foundations figure 1, projection-symmetry figure 2, dot-product geometry, projection-reference-directions) and ensure each kwarg maps to either the `VectorPanel` constructor or a Plottable.

### Risk 8: Mixed-type panel composition (Static + Dynamic + Interactive in one Figure)
- **Problem:** CONTEXT calls this the "primary goal" but allows "same-type-only fallback if mpl animation+widget interplay blocks mixing for >2 hours." matplotlib's `FuncAnimation` and ipywidgets event loops can conflict — DynamicPanel uses a timer-driven `FuncAnimation`, InteractivePanel uses widget callbacks. Co-existence in one `Figure` may require a shared frame-clock orchestrator at the Figure level.
- **Mitigation:** Implement same-type-only first (easier verification), then attempt mixed. Set a 2-hour budget on mixing per CONTEXT. Document the fallback decision in PLAN.md so the planner knows to gate mixed-type behind an explicit go/no-go check.

### Risk 9: 4 RETIRED helpers in dsp_figures.py are dead but their nested helpers (`_draw_proj` × 2 variants) inform the live `_draw_reconstruction` logic
- **Problem:** The retired functions contain `_draw_proj` (in two slightly different forms) — older versions of the projection-decomposition pattern. They're dead code but their existence might confuse the porter ("which is the canonical projection-draw?").
- **Mitigation:** Delete RETIRED helpers as part of the migration (rather than carrying them forward). The canonical is `_draw_reconstruction` from `_plot_projection_reconstruction_either_order:1267-1312`.

### Risk 10: `research/utilities/__init__.py` re-exports become inconsistent after split
- **Problem:** `__init__.py` (lines 64-82) re-exports every plotting helper. After moving most to dsplot, the re-exports either (a) re-route to dsplot for backwards compat, or (b) get dropped (breaking comparison.py / figures.py imports).
- **Mitigation:** Keep the re-exports but route through dsplot (e.g., `from dsplot.primitives import compute_freq_yticks as compute_freq_yticks`). Document the deprecation. Eventually remove when comparison.py/figures.py get refactored in a later phase.

---

## Coverage summary

| Category | Count | Files involved |
|---|---|---|
| Live figures to port | 7 (5 foundations + motivator + alignment) | dsp_figures.py |
| RETIRED helpers | 4 (xy_projection, similarity, projection, ref_directions, dot_product_symmetry) | dsp_figures.py |
| Private composition helpers to migrate | 3 (`_plot_sharp_vector`, `_panel_titles`, `_draw_dashed_tip_to_tail`) | dsp_figures.py |
| utilities/plotting.py functions to port | 8 of 17 (the rest stay or are out-of-scope) | utilities/plotting.py |
| utilities/plotting.py functions that stay | 9 (host-coupled, comparison-grid-only, or FFT-only) | utilities/plotting.py |
| Style constants to rename | ~25 of ~70 | utilities/style.py |
| Files moved entirely | 1 (style.py → dsplot/style.py with renames + shim) | utilities/style.py |
| Files split | 1 (plotting.py — some functions move, others stay) | utilities/plotting.py |
| Files untouched | 5 (constants, dsp_helpers, printing, signals, timing, wav_export) | utilities/ |
| New dsplot modules | ~13 (style + primitives + 8 plottables + 3 panels + figures/ subpackage) | dsplot/ |
