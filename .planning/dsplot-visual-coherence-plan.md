v# Plan — dsplot Visual Coherence: one style guide, smart cascade

## Goal
Make every dsplot figure (1, 2.4.1, 2.4.2, 2.4.3, 2.5, 2.6) derive its visual tone from **one
common style guide**, so changing a thing in one place updates it everywhere it *should* — and
nowhere it shouldn't. Keep per-figure overrides for genuinely unique needs. The cascade must hold
identically across the three render paths: **production PNG, GIF, and the live notebook (ipynb)**.

## Where We Are (plain English)

_Updated: 2026-06-24._

Two tracks are running in parallel:

1. **Library plumbing** (the phased plan below): we've laid the foundation so one
   style knob can cascade everywhere. **Done so far — Phase 0, 1, and 2a — with
   zero visual change** (proven pixel/byte-identical against the baseline oracle):
   - Phase 0: snapshotted every figure as a visual-regression baseline.
   - Phase 1: added named render profiles (`print` / `notebook` / `gif`).
   - Phase 2a: swept ~25 hardcoded numbers into named style constants.
   - **Next up: Phase 2b + Phase 3** — the FIRST steps that intentionally change
     how things look (bone-white chrome + promoting shared pieces into the
     library). These are held at a **review gate**: nothing changes until you see
     before/after.

2. **Per-figure cosmetic pass** (the review tracker below): going figure by
   figure, fixing the look, getting your eyeball approval, then publishing.
   Static/print look first; animation + GIF come after each figure's static look
   is locked.

### Figure status at a glance

| Figure | What it is | Status |
|---|---|---|
| **Fig 1** | Fourier vs Wavelet | ☐ todo |
| **2.4.1** | xy recombine | ✅ published (static) — `fig_2_4_1_xy_recombine_composite_v20.png` |
| **2.4.2** | a onto b | ☐ todo — next big build |
| **2.4.3** | dot product 3d | ☐ todo |
| **2.5** | sign accumulation | ☐ todo |
| **2.6** | sine basis | ☐ todo |

Legend: ☐ todo · ◐ rendered, awaiting review · ✅ approved & published.
Full per-figure notes are in the **Review Tracker** lower in this doc.

## Background / Context
`research/dsplot/style.py` is already a real central style module (palette, typography, layout
inches, `nb_compact_style()`). Most plottables (10/11) and most panels read it *lazily at draw
time*, so global changes cascade. The framework is good. The problem is **not** "no style system" —
it's that the system is **half-formalized**, so each figure has quietly grown its own copy of the
cross-cutting bits, and a pile of hardcoded literals sit outside the theme's reach.

Four audits (figures / panels / plottables / figure+axes_setup) found the concrete gaps below.

### What's already right (keep, don't rewrite)
- Central `style.py` with lazy resolution; `style.X` read at draw time.
- Shared decoration helpers: `_apply_axis_decoration`, `_pin_extreme_ticklabels`, `_render_chrome_titles`.
- Plottables follow a clean "param=None → resolve `style.*` in draw()" contract (`Dropline` is the gold standard).
- `nb_compact_style()` is the right *idea* for scale profiles.
- Figures 2.5/2.6 already model a disciplined 3-path render contract — we generalize *their* pattern.

> Constraint (CLAUDE.md): no rewrites. This plan **builds on** the module-global + lazy + context-manager
> pattern that already works. It does not replace it with a from-scratch OOP theme engine.

### The gaps (what breaks the "one knob updates everything" goal)

**G1 — No named render profiles.** Each figure hand-rolls its scale per path: ad-hoc `unit_inches` /
`unit_height_inches` / `dpi` args + a private stack of context managers
(`_unified_style`/`_static_chrome_style`/`_cell_border_chrome`). There is no single "this is the
PRINT profile / NOTEBOOK profile / GIF profile" object. DPI isn't even in `nb_compact_style`.

**G2 — Cross-figure duplication of library-level concepts:**
| Concept | Duplicated in | Should live in |
|---|---|---|
| `_cell_border_chrome` (spines off, cell border is frame) | 2.5, 2.6 | library (a border-model option/profile) |
| `_inset_ticks` (ticks point inward) | 2.5, 2.6 | a `style.DEFAULT_TICK_DIRECTION` + panel honor it |
| `_bolden_spines` | 2.5, 2.6 | shared util / border-model option |
| light-chrome palette `#EEEEEE` for tick/spine/suptitle | fig 1, 2.5, 2.6 | a real palette decision (see Decision B) |
| `_SignStrip` (+/−/0 glyph strip) | 2.5, 2.6 | `dsplot.plottables.SignStrip` |
| wrapped/top-anchored caption panel | 2.5, 2.6 | a `TextPanel` preset/factory |

**G3 — Hardcoded literals bypassing `style.py`** (can't be themed or cascaded):
- panels: tick length/width multipliers `*0.6`/`*0.8`, `direction="inout"/"out"`, `edge_clearance=0.15`,
  title `fontweight="bold"`, subtitle/caption `italic`+`bold`, TextPanel `min_font_size`/`cell_padding_frac`/`line_spacing`.
- `axes_setup.py`: vector-axis `alpha=0.85`, `linewidth=1.8`, arrow `mutation_scale=14`, label `+2` bump,
  label pos `lim*0.92`, result text `"monospace"` + offsets.
- plottables: `Stem` marker `"o"`/`6.0`, `Vector` 3D tip `s=80` + `+0.6` lw, `Spotlight` `s=100`/`edgecolors="white"`.
- `figure.py`: figure-number y `1.05`, caption y `0.75`, bottom-pad buffer `0.05`.

**G4 — Cascade-correctness bugs (updates that *don't* propagate where they should):**
- `SuptitlePanel` resolves style **eagerly in `__init__`** — the only panel that does; later style changes never reach it.
- Animation background chrome (`DynamicPanel._render_background`) is resolved **once**; bg/spine/grid/tick
  styling is frozen before the frame loop. Foreground plottables redraw lazily, background doesn't — so a
  theme change can split-update an animation.
- `nb_compact_style` only changes inches+fonts, **not DPI**, and silently requires compose+render to both sit
  inside the context (fragile ordering).
- `VectorComponents` droplines are hardcoded to `SPINE_COLOR`, ignoring the `dropline_color` param.

**G5 — 2D vs 3D divergence** ("clean 2d 3d"): 3D spines use `DEFAULT_VECTOR_LINEWIDTH` not
`DEFAULT_SPINE_LINEWIDTH`, a separate decoration path, hardcoded spine alpha/`box_zoom`/label bold-italic.

## Key Files / References
- `research/dsplot/style.py` — the style guide (extend here).
- `research/dsplot/figure.py` — layout + chrome + master animation clock.
- `research/dsplot/axes_setup.py` — vector-axis decoration (many hardcodes).
- `research/dsplot/panels/*.py` — `base`, `static_panel`, `static_panel_3d`, `dynamic_panel`,
  `heatmap_panel`, `time_series_panel`, `composite_panel`, `suptitle_panel`, `text_panel`.
- `research/dsplot/plottables/*.py` — `vector`, `vector_components`, `stem`, `spotlight`, `dropline` (reference).
- `research/dsplot/figures/gen_figure_{1,241,242,243,2_5,2_6}*.py` — the 6 consumers.

## The Plan (do this FIRST — proposed; no work until approved)

Phased + incremental (per CLAUDE.md test approach). Each phase is independently verifiable by
re-rendering all 6 figures × 3 paths and eyeballing against the canonical baselines in
`assets/images/dsp/figures/by_figure/`.

### Phase 0 — Baseline snapshot (no behavior change)
Render every figure's PNG + GIF + (where applicable) a notebook still, into a `by_figure/<fig>/_premerge/`
scratch set. This is the visual regression oracle for every later phase.

### Phase 1 — Named render profiles (addresses G1, G4-DPI)
Introduce **one** library concept for scale/path in `style.py`:
`render_profile("print" | "notebook" | "gif")` — a context manager that sets the full bundle
(fonts, inches, **dpi**, tick direction, border model). `nb_compact_style()` becomes the `"notebook"`
profile (kept as an alias for back-compat). Each figure's `render`/`save_gif`/`show` selects a profile
instead of hand-passing `unit_inches`/`dpi`. Per-figure unique sizing stays as explicit args layered
on top. **Decision A** governs how far this goes.

### Phase 2 — Sweep hardcoded literals into `style.py` (G3)
Add named constants for every literal in G3 and route the call sites through them. Add the scale-sensitive
ones to the profile overrides so they actually rescale in notebook/gif. Pure mechanical; baseline-identical
in print, and *more* correct in notebook.

### Phase 3 — Promote duplicated concepts to the library (G2)
- `style.DEFAULT_TICK_DIRECTION` honored by panels → deletes `_inset_ticks` copies.
- A **border model** option ("spine" vs "cell-border") on Figure/profile → deletes `_cell_border_chrome` +
  `_bolden_spines` copies.
- New `dsplot.plottables.SignStrip` → deletes the two `_SignStrip` copies.
- A `TextPanel` "caption" preset (wrapped, top-anchored, borderless) → deletes the duplicate caption builders.
Figures shrink to *content*; shared look comes from the library.

### Phase 4 — Cascade-correctness fixes (G4)
- Make `SuptitlePanel` resolve lazily at render (match every other panel).
- Re-apply background chrome inside the animation tick (or document + guarantee profile-before-render) so
  animations cascade like statics.
- Fix `VectorComponents` dropline to honor `dropline_color`.

### Phase 5 — 2D/3D unification (G5)
Route 3D spine/tick/label chrome through the same style constants + a shared decoration contract; move
3D-only knobs (spine alpha, `box_zoom`, 3D linewidth offset) into `style.py`. Goal: a 2D and a 3D panel
in the same figure read as one system.

### Phase 6 — Document the contract
Short "dsplot style guide" doc: the palette roles, the profile system, the override ladder
(global → profile → figure-local → per-object), and the rule "new figures select a profile + use library
plottables; only override for genuinely unique needs."

## How to Run It / See Results
- Regenerate all: `source venv/bin/activate && python -m research.dsplot.figures` (PNGs) + the per-figure
  `save_gif()` entrypoints + `/tmp` gif driver used for 2.4.x.
- Notebook path: `src/subshader/dsp/dsp.ipynb` (needs `%matplotlib widget` first — see memory).
- Compare each new render against `assets/images/dsp/figures/by_figure/<fig>/` canonicals. New iterations get
  new `_vNN` names (never overwrite — mobile caches by URL).

## Confirmation / Validation
After each phase I'll re-render the affected figures across all three paths and show you before/after so you
confirm the look is unchanged (Phases 0/2) or improved (Phases 1/3/4/5) before moving on. Nothing merges
without your eyeball.

## Decisions (LOCKED)
- **A — Theme mechanism:** Formalize the proven pattern — named `render_profile()` + constants in `style.py`.
  No new object model, no figure rewrites. Changing a constant/profile cascades everywhere via the existing
  lazy lookup.
- **B — Chrome palette:** Bone-white `#EEEEEE` (= the palette's existing `NEUTRAL_COLOR`) becomes the shared
  default for chrome roles (`TICK_LABEL_COLOR`, `SPINE_COLOR`, `SUPTITLE_COLOR`, and `DROPLINE_COLOR` to review).
  The three figures (1, 2.5, 2.6) that currently hand-set `#EEEEEE` drop that local override. NOTE: `SPINE_COLOR`
  moves `#444444 → #EEEEEE`, a large tonal shift wherever spines are visible — this is exactly what the Phase
  review gate is for (2.5/2.6 already hide spines behind the cell border, so impact is mostly fig 1 + vector axes).
- **C — Sequencing:** Phased, with a review gate after each phase (re-render 6 figures × 3 paths, eyeball vs
  `by_figure/` canonicals, confirm before next phase).

## Work Log / Tracking
- What we did: assessed dsplot style architecture via 4 parallel audits; wrote this plan.
- What we learned: system is half-formalized — good lazy core, but per-figure duplication + hardcodes + a few
  cascade bugs (eager Suptitle, frozen animation background, DPI-less notebook profile).
- What did NOT work: n/a (assessment only).
- Decisions made: A=formalize proven pattern; B=bone-white #EEEEEE (NEUTRAL_COLOR) as default chrome; C=phased w/ review gate.

### Round 1 — Phases 0, 1, 2a (2026-06-23)
- **Restore point**: commit `96936d0` snapshots the whole working tree before any dsplot edits (the one
  pre-authorized commit). `git reset --hard 96936d0` returns to "here".
- **Phase 0 — baseline oracle**: rendered all 6 figures × applicable paths (8 PNG + 5 GIF = 13 artifacts)
  into `assets/images/dsp/figures/by_figure/<fig>/_premerge/` via `/tmp/phase0_baseline.py`
  (driver takes a subdir arg; supports both dsplot import worldviews). This `_premerge/` set is the
  permanent visual-regression oracle — kept in tree, do NOT delete.
- **Phase 1 — render profiles**: added `render_profile("print"|"notebook"|"gif")` + `PROFILES` registry to
  `style.py`; `nb_compact_style()` is now a byte-identical alias for `render_profile("notebook")`.
  Exported `render_profile` from `dsplot/__init__`. Verified: re-render diffed vs oracle → all PNGs
  maxdiff=0, all GIFs byte-identical. ZERO behavior change (mechanism only).
  - DPI fold-in deferred: notebook dpi legitimately varies per figure (72 vs 80), so it's NOT a single
    notebook-profile value. Print dpi stays `DEFAULT_DPI=150`. A canonical gif dpi can fold into the "gif"
    profile during figure migration (later, gated).
- **Phase 2a — literal sweep (zero-change half of Phase 2)**: extracted ~25 hardcoded literals into named
  `style.py` constants AT CURRENT VALUES and routed call sites (axes_setup, figure.py, panels/base,
  static_panel, dynamic_panel, heatmap_panel, text_panel, plottables/stem, spotlight, vector). New constant
  families: chrome emphasis (title/subtitle/caption/figure weight+style), tick decoration
  (DEFAULT_TICK_DIRECTION + inset scales + heatmap direction/clearance), vector-axes decoration weights +
  label glyphs + result readout, plottable markers (stem/spotlight/3D vector tip), text-panel sizing,
  figure-chrome positions. stem marker/markersize and text-panel sizing converted to None→lazy-resolve.
  3D panel literals deliberately LEFT for Phase 5. Verified: all PNGs maxdiff=0, all GIFs byte-identical.
- **What worked / learned**: the oracle + parametrized driver makes "zero visual change" provable, not
  asserted. Both Phase 1 and 2a are pixel/byte-identical across all 13 artifacts.
- **Next**: **Phase 2b** (add scale-sensitive new constants to the notebook/gif profile overrides so they
  rescale — intentionally changes notebook/gif output) and **Phase 3** (bone-white `#EEEEEE` chrome shift
  per Decision B + promote duplicated concepts). These are the FIRST intentional-visual-change steps →
  REVIEW GATE: present before/after to user before applying. Holding here.
- **Note**: nothing committed beyond the restore point (per user directive "no commits except to get back
  to here"). All Phase 1/2a edits live uncommitted in the working tree.


---

# dsplot Visual-Coherence Review Tracker

_Living status board for the per-figure consistency pass. Updated: 2026-06-24._

Companion to `dsplot-visual-coherence-plan.md`. This file tracks **what we're
reviewing and the most recent update per item**.

Status legend: ☐ todo · ◐ in progress (rendered, awaiting review) · ✅ approved & published

---

## Locked decisions

- **Chrome:** bone-white `#EEEEEE` unified text; SPINE off → the single gray
  cell border (NEUTRAL `#EEEEEE`) is each panel's frame (the 2.5/2.6 model).
- **Title convention (from fig 1):** figure NAME in the top suptitle band,
  "Figure N" in the bottom footer band.
- **2.4.2 / 2.4.3 layout:** two stacked 2.5-style rows — cosine form on top,
  component form below; each `[ visual | AccumulatorStrip | text box ]`; result
  strips vertically aligned so "identical" reads as two equal bars.
- **Strip name:** `AccumulatorStrip` (lifted from 2.5's `_AccumStem` +
  `_SumReadout` into the library; shared by 2.4.2 / 2.4.3 / 2.5 / 2.6).
- **Reference b:** keep magnitude 3; **both** result strips display `a·b` (the
  dot product), so the bars match without distorting the geometry.
- **Publishing:** by_figure convention `{slug}_composite_v{N}.png`; prior live
  version moves to `archive/`. No commits (one restore point already taken).
- **Scope order:** static/print cosmetics first; dynamic (notebook) + GIF paths
  deferred until the static look is locked per figure.

---

## Per-figure

### Figure 1 — Fourier vs Wavelet
- **Notes:** widen the plot panels horizontally (more aspect vs the text boxes);
  use more time series / spread oscillations so they read less cluttered.
- **Status:** ☐
- **Recent:** —

### 2.4.1 — xy recombine
- **Notes:** name in title; consistent font color; plots fill their panels; drop
  the a/b/c sub-labels; descriptive names as per-panel footers.
- **Status:** ✅ published `fig_2_4_1_xy_recombine_composite_v20.png` (static path)
- **Recent (06-23→24):** bone-white unified chrome; tight spacing (pad 1.5→0.4″,
  gutter 3→0.6″) so plots fill; descriptive footers at 16pt; a/b/c removed; name
  title + "Figure 2.4.1" footer; old `reverify_v19` → `archive/`.
- **Open:** lower-quadrant emptiness left as-is (Q1 data on centered axes);
  dynamic/notebook path not yet updated.

### 2.4.2 — a onto b  ← next big build
- **Notes:** font color; text boxes too spacious; title like fig 1; redesign to
  two 2.5-style rows (cosine row + component row); AccumulatorStrip shows `a·b`
  in both, bars equal; maybe squeeze the arrow plots in with the strip.
- **Status:** ☐
- **Recent:** —

### 2.4.3 — dot product 3d
- **Notes:** copy 2.4.2 style; same spacing/name treatment; fix the fuzzy/small
  3D arrows (zoom the camera in + larger / higher-res arrows).
- **Status:** ☐
- **Recent:** —

### 2.5 — sign accumulation
- **Notes:** looks clean; generalize the Sum strip → `AccumulatorStrip` in the
  library without changing 2.5's look much; fix text-box margins.
- **Status:** ☐
- **Recent:** —

### 2.6 — sine basis
- **Notes:** move "purple reference = 1 Hz" into the text box; bring fig-1-style
  language into the text boxes.
- **Status:** ☐
- **Recent:** —

---

## Foundation / shared (cross-cutting)

### Round A — done (2026-06-24, library-first, cascades to all 6)
- ✅ **One sans font everywhere.** Added `style.DEFAULT_FONT_FAMILY = "DejaVu Sans"`;
  applied globally via rcParams in `Figure.__init__`. Removed `fontfamily="Ubuntu"`
  hardcodes in 2.4.2 / 2.4.3 (the cross-figure clash). Numeric readouts route
  through `DEFAULT_MONO_FONT_FAMILY`. One knob to change the whole type system.
- ✅ **Visible thin frame model.** Added `style.DEFAULT_FRAME_COLOR / _LINEWIDTH /
  _ALPHA`; `figure.py _draw_cell_borders` now reads them. The cell border is the
  single crisp frame on every plot (was a faint debug layer). Resolves "can't see
  the spine" at the library level.
- Render check: `/tmp/render_roundA.py` → `_review/_roundA/montage_roundA.png`.

### Round B — per-figure layout (in progress, 2026-06-24)
- ◐ **2.4.1** — rendered, awaiting review. Header consolidated to ONE band
  ("Figure 2.4.1 - Basic Vector Projection", 28pt, the 2.5/2.6 convention);
  giant bottom "Figure 2.4.1" footer band dropped → plots get the space.
  In-plot x/y glyphs 34pt → 22pt (via `DEFAULT_AXIS_LABEL_SIZE` in
  `_STATIC_CHROME`). Frame now visible (Round A cell border). Per-panel
  descriptive footers kept. Empty lower half left as-is (StaticPanel forces a
  symmetric square axis — asymmetric-ylim fill is separate surgery; 2.4.2 has
  the same centered-origin look). Render: `_review/_roundB/f241_roundB.png`.
  NOT yet published (pending review → then v21, archive v20).
- ◐ **2.4.2** — STATIC redesign rendered, awaiting review. Two stacked rows
  [visual | a·b strip | text]: cosine row (a, b, θ, projection shadow) + component
  row (a, b ghosted + x/y staircase); both `AccumulatorStrip(a·b=9)` so bars align
  vertically. One header band ("Figure 2.4.2 - Projection onto Another Vector"),
  bone-white chrome, square visual cells, RHS 2.5-style text boxes (SCAFFOLD copy —
  user authors final). Fixed pose a=(2,3) b=(3,1). Dynamic/notebook path UNCHANGED
  (deferred). Render: `_review/_roundB/f242_roundB.png`. Open: centered-origin
  empty lower-half (StaticPanel symmetric-axis limit); strip cell slightly wide.
- ☐ 2.4.3 / 2.5 / 2.6 / fig-1 hero+antihero — queued (see Per-figure section).

### Still to extract
- ✅ **`AccumulatorStrip`** extracted → `dsplot/plottables/accumulator_strip.py`
  (zero line + heavy stem + big readout; from 2.5's `_AccumStem`/`_SumReadout`).
  Exported as `dsplot.AccumulatorStrip`; themeable via `style.DEFAULT_ACCUM_*`.
  Smoke-tested pos/neg. 2.5 NOT yet migrated onto it (do later, verify identical).
- ☐ Extract `SignStrip` → library
- ☐ Promote chrome helpers (`_unified_style` / `_cell_border_chrome` /
  `_uniform_borders` / `_inset_ticks`) → library. 2.4.1 currently carries a local
  `_static_chrome` copy to fold in later.
- ☐ Port `_publish_path` auto-versioning into 2.4.1 / 2.4.2 / 2.4.3 `render()`
  (publishing is manual for now).
- ☐ Dynamic (notebook) + GIF paths — deferred per scope order.

---

## Render / verify harness

- `/tmp/phase0_baseline.py` → 13 artifacts; `_premerge/` snapshot = visual oracle
  (pixel-diff PNGs, byte-size GIFs).
- Composite montage: `/tmp/montage_all.py` → `/tmp/dsplot_all_figures_composite.png`.
