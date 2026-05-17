# Phase 9: dsplot Library Architecture — Context

**Gathered:** 2026-05-17
**Status:** Ready for planning
**Source:** Inline discussion with user (locked decisions captured below)

<domain>
## Phase Boundary

Phase 9 delivers a **standalone, project-agnostic Python plotting library** named `dsplot` that lives at `research/dsplot/` in this repo. The library has zero imports from `src/subshader/*` — SubShader at runtime never sees it. Consumers are: the `src/subshader/dsp/dsp.ipynb` notebook (animated figure rendering), README figure generation (static export), test suites (programmatic result plotting), and future external projects (importable as a standalone package).

The phase replaces the monolithic `research/dsp_figures.py` (and the loose helpers in `research/utilities/`) with a structured OOP module organized around two layers: **Plottables** (units drawn onto an axes) and **Panels** (containers that own an mpl Axes and orchestrate static/animated/interactive rendering).

</domain>

<decisions>
## Implementation Decisions

### Library Identity (LOCKED)
- **Name:** `dsplot`
- **Module home:** `research/dsplot/` (standalone Python package; zero imports from `src/subshader/*`)
- **Library boundary (LOCKED — D-01):** the library proper is `research/dsplot/` MINUS `research/dsplot/figures/`. The `figures/` subdir is CONSUMER code (it generates the canonical DSP.md PNGs) and is allowed to import from `research.utilities` for CWT computation. The isolation grep MUST use `--exclude-dir=figures`:
  ```
  grep -rIn "subshader" research/dsplot/ --exclude-dir=figures
  ```
- **Extractability:** Design so the directory can be lifted out as its own pip-installable package later. No path dependencies on the host repo's layout beyond import-time discovery.
- **Style configurability (LOCKED — D-05, REDESIGNED):** Module-level constants in `dsplot.style` (palette, fonts, sizes, **layout/format defaults**) are an INHERITED TEMPLATE. Every figure inherits them by reference; figures CAN locally override by introducing a module-level constant derived from the default (e.g. in `motivator.py`: `LABEL_RATIO = style.DEFAULT_LABEL_RATIO * 1.5`). Overriding the default GLOBALLY is also supported via simple reassignment (`dsplot.style.PRIMARY_COLOR = "#new"`) — affects every figure rendered after the reassignment. No config file, no plugin system, no DSL.

### Architecture — Two Layers (LOCKED)

**Layer 1 — Plottables** (units drawn onto axes):
- `Vector` — single arrow with origin, head, label. **Polymorphic on tuple length (LOCKED — D-02):** 2-tuple input → 2D rendering (mpl Axes); 3-tuple input → 3D rendering (requires an `Axes3D`, e.g. axes created with `projection="3d"`). Same API surface (origin, color, label, linewidth, alpha, linestyle, etc.). Dimensionality inferred from input. There is NO separate `Vector3D` class. **Asymmetric dispatch (LOCKED — D-06):** a 3-tuple drawn onto a 2D Axes raises TypeError (no defensible 3D→2D collapse); a 2-tuple drawn onto an Axes3D is implicitly extended to `(x, y, 0)` and rendered flat in the z=0 plane (defensible projection default — easier ergonomics for mixing 2D inputs into 3D scenes).
- `VectorComponents` — x/y decomposition of a vector (dashed component arrows + droplines)
- `TimeSeries` — 1D signal plotted vs time
- `Heatmap` — 2D array as colored image
- `Spotlight` — highlight overlay (rectangle, scatter, glow) for emphasizing a value/region
- `Dropline` — dashed perpendicular indicator from one point to another
- `Annotation` — text + optional arrow callout

Each Plottable has style knobs (color, linewidth, linestyle, alpha, label, z-order) — settable at construction (kwargs) and mutable after (attribute assignment). When a style knob is `None` at construction, it resolves against `dsplot.style.*` at `draw()` time (lazy lookup — what makes runtime style reassignment work).

**Layer 2 — Panels** (containers; each owns one mpl Axes):
- `StaticPanel` — single state, just renders
- `DynamicPanel` — `frames: list[State]`; advances on timer via mpl `FuncAnimation`; loops with auto-reset
- `InteractivePanel` — same frame model; advances on `matplotlib.widgets` controls (next/prev buttons always present; slider and checkbox optional). **AMENDED — D-07 (post-execution):** original LOCKED choice was `ipywidgets`; pivoted to `matplotlib.widgets` after user verification of 09-04. Rationale: the widgets render in inset axes anchored to the panel's `transAxes`, so controls live INSIDE the figure canvas and stay column-aligned under their panel in multi-panel Figures. Side benefit: drops the runtime ipywidgets dependency from the library proper.

**Composition:** `panel.add(plottable)` stacks elements on the same axes. Overlays are free (e.g., a `TimeSeries` plus a `Spotlight` on the same panel; or a `Heatmap` under a `TimeSeries`).

**Mixed-type Figures:** Primary goal — a single `Figure` can compose `StaticPanel`s, `DynamicPanel`s, and `InteractivePanel`s side by side. Same-type-only fallback (one Figure = all dynamic, or all interactive) is acceptable if mpl animation+widget interplay blocks mixing for >2 hours of work.

**3D axes support (LOCKED — D-04):** `Figure.add_panel(panel, *, row, col, rowspan=1, colspan=1, projection: str | None = None)` accepts a `projection` kwarg. When `projection="3d"`, the gridspec cell is created via `add_subplot(..., projection="3d")` and the panel receives an `Axes3D`. Panels that accept 3D axes must do their own 2D-vs-3D branching (or be subclassed — e.g. `StaticPanel3D` for the 3D foundation figure).

### Backend (LOCKED, with D-07 amendment)
- **matplotlib** for static + dynamic (`FuncAnimation`)
- **`matplotlib.widgets`** for interactive controls (D-07 amendment to original ipywidgets choice — see Layer 2 above)
- No other backends (no plotly, bokeh, etc.)

### Consumers (LOCKED — the library serves all four; library itself depends on none of them)
1. `src/subshader/dsp/dsp.ipynb` — notebook cell 01 renders figure 1 as a `DynamicPanel` (5-frame loop: vector `a` stays anchored at A=(2,3) across every frame, sibling `a'` cycles its x-component through [+2, +1, 0, -1, -2] while `a'.y` stays fixed at 3 — orthogonality made visible)
2. README figure generation — static PNG/GIF export
3. Test suites (unit + timing) — **one** example test showing the integration pattern; not full coverage
4. Future external projects — importable as standalone

### First Delivery Scope (LOCKED)
Full port of everything in `research/dsp_figures.py` into the new Panel/Plottable structure:
- `motivator` chirp (uses `TimeSeries` Plottable)
- `alignment_diagnostic` (uses `Heatmap` Plottable)
- `vector_basics`, `components_recombine_either_order` (figure 1), `projection_reconstruction_either_order` (figure 2), `dot_product_geometry`, `vector_projection_3d` (foundation figures — use `Vector` (polymorphic 2D/3D) and `VectorComponents` Plottables)
- `vector_xy_reconstruction` (LOCKED — D-03): regenerated via dsplot's `components_recombine.render_vector_xy_reconstruction(...)`. The historical orphan PNG is killed and replaced by a real reproducible render.

Constants `A`, `A_PRIME`, `B`, `A_Z`, `FOUND_LIM` currently in `dsp_figures.py` migrate alongside as figure-defining values (not library-level style constants).

`src/subshader/dsp/DSP.md` alt-text and LaTeX block already mirror these constants — any value change during migration must be flagged for the user to mirror into DSP.md.

### Code Quality Expectations
- **Self-documenting** via good naming and structure — minimal comments
- **OOP with clear single-responsibility classes** — each Panel type, each Plottable type
- **Modular:** any Panel works with any Plottable; any Plottable works on any Panel
- **Consistent visual tone** across all output (one `style.py` drives every figure as an inheritable template)

### Style as Inheritable Template (LOCKED — D-05)

`dsplot.style` is the SINGLE SOURCE OF TRUTH for every visual default — colors, typography, panel size, row layout, hspace/wspace, label ratios, margins, etc. Every figure consumes these defaults; figures that need different values reassign LOCALLY at the module level (deriving from the default to keep the intent traceable).

Constant naming convention:
- Generic, project-agnostic, role-named: `PRIMARY_COLOR`, `DEFAULT_HSPACE`, `DEFAULT_LABEL_RATIO`, `DEFAULT_PANEL_MARGIN`, `DEFAULT_TITLE_FONT_SIZE`, `DEFAULT_PANEL_SIZE_INCHES`, etc.
- NOT figure-specific: no `MOTIVATOR_*`, `GRID_*`, `WAVELET_*` names live in `dsplot.style`.
- Figure-local overrides live in the figure's own module: e.g. in `figures/motivator.py`:
  ```python
  from dsplot import style
  LABEL_RATIO = style.DEFAULT_LABEL_RATIO * 1.5   # motivator wants a chunkier label gutter
  HSPACE      = style.DEFAULT_HSPACE              # motivator inherits the global spacing
  ```

The two override modes both work:
- **Global reassignment:** `dsplot.style.PRIMARY_COLOR = "#new"` — affects every figure rendered after.
- **Local override:** a figure module reassigns its OWN module-level constant without touching `dsplot.style.*` — affects only that figure.

### Claude's Discretion
- Internal directory structure under `research/dsplot/` (subpackages, file splits)
- Exact constructor signatures for each Plottable and Panel
- Render-cycle implementation details for Dynamic/Interactive (single shared FuncAnimation orchestrator vs per-panel animations, widget event wiring strategy)
- Internal grouping inside `dsplot.style` (single `style.py` vs split into `style/colors.py` + `style/typography.py` + `style/layouts.py`) — as long as `from dsplot import style; style.PRIMARY_COLOR` works regardless
- Testing strategy for visual-fidelity verification (image hashing? side-by-side regenerate-and-diff?)

</decisions>

<non_goals>
## Non-Goals (anti-rabbit-hole guardrails — enforce during planning AND execution)

1. **No subshader imports anywhere in `dsplot` library proper.** `research/dsplot/figures/` is consumer code, allowed to import from `research.utilities` for CWT computation. Library boundary verification: `grep -rIn "subshader" research/dsplot/ --exclude-dir=figures` returns zero lines.
2. **No new plot types beyond the 7 Plottables + 5 concrete Panel uses catalogued.** Vector is polymorphic on input shape (2D/3D) — that's a dimensional extension of an existing type, not a new type. Polish before breadth.
3. **No CLI, no config file, no plugin system.** Style override is "import + reassign constants" (globally on `dsplot.style.*`, or locally in a figure module).
4. **No backends other than matplotlib.** Plotly/bokeh hooks are not Phase 9 work.
5. **No subshader-specific styling baked in.** Style constants are generic (e.g. `PRIMARY_COLOR`, not `WAVELET_ORANGE`; `DEFAULT_HSPACE`, not `MOTIVATOR_HSPACE`).
6. **Mixed-type composition is the goal, but same-type-only fallback is acceptable.** If mpl animation + widget interplay blocks mixing for >2 hours, ship same-type-only and defer.
7. **Test-suite plot helpers are a stub example**, not full coverage.
8. **No DSL / no config-driven figure assembly** — composition is just Python: instantiate, `.add()`, `.show()`.
9. **No "smart" auto-layout** beyond matplotlib defaults. If a layout needs tweaking, user passes positions.

</non_goals>

<canonical_refs>
## Canonical References

**Downstream agents (planner, executor) MUST read these before planning or implementing.**

### Existing code to port / extract from
- `research/dsp_figures.py` — the monolithic figure module being decomposed. All dispatched figures here become Panel-based equivalents.
- `research/utilities/style.py` — current style constants. Becomes the template for `dsplot.style`.
- `research/utilities/plotting.py` — current low-level plot helpers (`setup_vector_axes`, `create_panel_row`, `plot_vector`). These migrate into `dsplot.primitives` or get absorbed into Panel/Plottable methods.
- `research/utilities/printing.py`, `research/utilities/dsp_helpers.py`, `research/utilities/signals.py`, `research/utilities/wav_export.py`, `research/utilities/timing.py` — review each; some belong in `dsplot`, some stay in `research/utilities/` (utilities not related to plotting).

### Output target
- `src/subshader/dsp/dsp.ipynb` — cell 01 (currently empty code cell) is the demo target for the animated figure 1.
- `src/subshader/dsp/DSP.md` — references the figures by file path. Migration must keep filenames stable OR update DSP.md alt-text refs in lockstep.

### Memory / process constraints
- `feedback_remote_session_push_policy.md` — md/png pushes OK without confirmation; code changes need explicit user confirmation.
- `feedback_authoring_momentum.md` — surface scope creep rather than chasing detours.
- `feedback_png_naming.md` — never overwrite existing PNGs; new iterations get new versioned filenames.
- `feedback_collaborative_authoring_patterns.md` — for any user-facing prose (READMEs, docstrings the user will publish): candidate suggestions, structured skeletons, structural-pause checkpoints. (Library code docstrings are Claude's call.)

</canonical_refs>

<specifics>
## Specific Ideas

- **Example use case for `DynamicPanel`:** Vector projection animation that varies the x-component across frames while y-component stays fixed, visually demonstrating orthogonality (the y measurement is unaffected by changing x).
- **Example use case for `InteractivePanel`:** "Spotlight" cycling through array values — user presses Next to advance which value is highlighted; useful for stepping through DSP examples too fast to follow as animation.
- **Example use case for overlays:** Instantaneous frequency curve overlaid on a time series; or a wavelet coefficient heatmap with a time series rendered above it.

</specifics>

<verification_target>
## Verification Target (Phase Completion Criteria)

1. **Module isolation:** `grep -rIn "from subshader" research/dsplot/ --exclude-dir=figures` returns zero results. `grep -rIn "import subshader" research/dsplot/ --exclude-dir=figures` returns zero results. (Per D-01: the `figures/` subdir is consumer code, explicitly excluded from the library boundary.)
2. **Figure parity:** Every figure currently produced by `research/dsp_figures.py` is reproducible from `dsplot` with identical or better visual fidelity. (Side-by-side comparison or image-diff check.) Includes `vector_xy_reconstruction.png` regenerated via dsplot (D-03 — orphan retired).
3. **Notebook demo:** `src/subshader/dsp/dsp.ipynb` cell 01 imports from `dsplot` and renders figure 1 as a `DynamicPanel` showing vector `a` ANCHORED at A=(2.0, 3.0) across every frame (fixed reference) while sibling vector `a'` SWEEPS its x-component through 5 values [+2.0, +1.0, 0.0, -1.0, -2.0] one per frame, with `a'.y` held constant at 3.0 throughout. The orthogonality beat reads as: a' moves left-right across the y=3 horizontal while its height never changes — y is invariant under changes in x. Animation loops with auto-reset (after frame 4, returns to frame 0).
4. **Test integration:** At least one unit test (e.g., a new `test_dsplot_example.py` in `research/tests/`) imports `dsplot`, runs a Plottable+Panel composition, and saves a result PNG. Demonstrates the testing-consumer pattern.
5. **Style override works:** A short demonstration snippet (in the phase summary or a doc cell) shows BOTH override modes:
   (a) **global reassignment** — `dsplot.style.PRIMARY_COLOR = "#new"` then re-rendering produces every-figure-affected output;
   (b) **local figure override** — a single figure module redefines its own derived constant without touching `dsplot.style.*`, and the global default is unaffected for other figures.
6. **Old module retired:** `research/dsp_figures.py` is either deleted or reduced to a deprecation shim that re-exports from `dsplot` for backwards compat during the transition.

</verification_target>

<deferred>
## Deferred Ideas

- Plotly / Bokeh / other backend support (explicitly out per Non-Goal #4)
- Cross-project pip packaging (the lib is *designed* to be extractable but actual extraction to its own repo / PyPI is post-Phase-9)
- Full test coverage for the lib (one example stub is sufficient; comprehensive test suite is a future phase)
- DSL or config-driven figure assembly (explicitly out per Non-Goal #8)
- Smart auto-layout (explicitly out per Non-Goal #9)
- Separate `Vector3D` class (REJECTED — D-02: polymorphic Vector handles 2D/3D via tuple-length dispatch)

</deferred>

---

*Phase: 09-dsplot-library-architecture*
*Context gathered: 2026-05-17 via inline discussion (no formal /gsd:discuss-phase invoked — decisions captured directly in conversation and consolidated here)*
*Revised: 2026-05-17 — applied 5 locked decisions (D-01 boundary, D-02 polymorphic Vector, D-03 orphan retire, D-04 3D projection kwarg, D-05 style template with override). Second revision: D-06 asymmetric Vector dispatch (2D→3D implicit, 3D→2D strict). Third revision (post-execution): D-07 amendment — InteractivePanel uses `matplotlib.widgets` (not ipywidgets) for prev/next/slider/checkbox controls. Anchored inset_axes keep controls inside the figure canvas, column-aligned under their panel; drops the runtime ipywidgets dependency. Amendment recorded after user verification of 09-04 smoke tests including the LOCKED primary mixed-type goal (Dynamic + Interactive on one Figure).*
