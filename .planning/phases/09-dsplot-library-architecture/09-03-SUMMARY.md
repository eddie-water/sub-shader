---
phase: 09-dsplot-library-architecture
plan: 03
subsystem: dsplot (2D Plottables — completion of the 7-Plottable catalog)

tags: [dsplot, plottable, time-series, heatmap, spotlight, dropline, freq-axis, d-05]

# Dependency graph
requires:
  - phase: 09-dsplot-library-architecture / plan 01
    provides: Plottable abstract base + .draw(ax) contract, dsplot.style template
  - phase: 09-dsplot-library-architecture / plan 02
    provides: Panel ABC + StaticPanel + Figure orchestrator (consumers of these Plottables)
provides:
  - dsplot.TimeSeries — 1D signal vs time via fill_between (motivator §1 figure)
  - dsplot.Heatmap — 2D imshow with log_freq y-tick option (CWT/STFT spectrograms)
  - dsplot.Spotlight — 3-mode highlight overlay (rectangle / scatter / glow)
  - dsplot.Dropline — dashed perpendicular indicator between two points
  - dsplot.freq_axis.compute_freq_yticks — log-frequency tick computation helper
affects:
  - 09-04-3d-foundation (no impact — 3D rendering uses the polymorphic Vector from 09-01)
  - 09-05-figure-ports (Wave 3 ports can now compose every dsp_figures.py output from public dsplot API)
  - 09-06-notebook-demo (motivator-style DynamicPanel demos have all needed Plottables)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy style lookup at draw() (D-05): all four new Plottables resolve None style knobs against dsplot.style.* INSIDE draw(), so runtime style reassignment between construction and draw is observable. Pinned by 2 dedicated tests (Heatmap.DEFAULT_HEATMAP_CMAP, Dropline.DROPLINE_COLOR)."
    - "Defer validation to draw(): Spotlight's mode/range checks raise ValueError on .draw(), not in __init__ — matches the rest of the library's contract (a misconfigured Plottable can be constructed and replaced before render without immediate failure)."
    - "Default-color role assignment for non-identity Plottables: TimeSeries → NEUTRAL_COLOR (the palette reserves PRIMARY/SECONDARY for vector identity; bare data display is non-identity); Spotlight → TERTIARY_COLOR (gold, high visibility against dark bg); Dropline → DROPLINE_COLOR (dedicated palette slot)."
    - "Shadow-attribute pattern for tri-state defaults: Dropline shadows raw kwargs as _raw_alpha / _raw_linestyle so the base Plottable's fixed defaults (alpha=1.0, linestyle='-') don't shadow the None-detection signal. Keeps lazy lookup working even when the base class fixes a default value."

key-files:
  created:
    - research/dsplot/freq_axis.py
    - research/dsplot/plottables/time_series.py
    - research/dsplot/plottables/heatmap.py
    - research/dsplot/plottables/spotlight.py
    - research/dsplot/plottables/dropline.py
    - research/tests/dsplot/test_time_series_heatmap.py
    - research/tests/dsplot/test_spotlight_dropline.py
  modified:
    - research/dsplot/__init__.py
    - research/dsplot/plottables/__init__.py

decisions:
  - "TimeSeries default color = NEUTRAL_COLOR (not PRIMARY): a bare TimeSeries is generic data display, not a vector identity. PRIMARY / SECONDARY remain reserved for vector-identity Plottables (Vector, VectorComponents). Documented in TimeSeries class docstring."
  - "Spotlight default color = TERTIARY_COLOR (gold): the palette's high-visibility accent slot, chosen to stand out as a foreground emphasis layer above any data Plottable. Tested implicitly through the artist-existence tests."
  - "Heatmap vmax precedence: explicit vmax > vmax_percentile kwarg > style.DEFAULT_HEATMAP_VMAX_PERCENTILE. Pinned by test 7 (test_heatmap_vmax_percentile_resolves_to_np_percentile). Easy to invert by accident — the test guards against regression."
  - "Spotlight mode='rectangle' supports BOTH x_range AND y_range simultaneously (renders two crossed axspans). Locked by Test 3. Rationale: alignment-diagnostic-style markers often need a time band AND a frequency band crossed; making consumers compose two Spotlights would be ceremony for a common pattern."
  - "Dropline uses _raw_alpha / _raw_linestyle shadow attributes: the Plottable base class fixes alpha=1.0 and linestyle='-' as concrete defaults, but Dropline needs to distinguish 'user didn't pass alpha' (resolve at draw) from 'user passed alpha=1.0' (use it). Solved by shadowing the raw kwargs as instance attrs and checking them at draw(). Considered changing the base class but rejected — would ripple into 5 existing Plottables for one consumer's need."

patterns-established:
  - "lazy-style-lookup uniform across all four new Plottables (D-05 enforced)"
  - "draw-time validation (mode / required-kwarg checks raise ValueError in .draw, not __init__)"
  - "non-identity-color = NEUTRAL palette slot (TimeSeries documented; Heatmap implicit via cmap default)"

requirements-completed: [PLOT-09, PLOT-10, PLOT-11, PLOT-12, PLOT-14]

# Metrics
duration: ~8min
completed: 2026-05-17
---

# Phase 09 Plan 03: Remaining 2D Plottables + freq_axis Helper Summary

**Closes the 7-Plottable catalog with TimeSeries, Heatmap, Spotlight, Dropline, and the freq_axis log-frequency tick helper. All four Plottables conform to the 09-01 Plottable contract with lazy style lookup at draw() time per D-05. Zero subshader imports in the library proper.**

## Performance

- **Duration:** ~8 min (4 task commits, 2 RED + 2 GREEN, full TDD compliance)
- **Started:** 2026-05-17T17:57:20Z
- **Completed:** 2026-05-17T18:04:55Z
- **Tasks:** 2 (both TDD)
- **Files created:** 7 (5 library + 2 test files)
- **Files modified:** 2 (__init__.py re-exports)
- **Tests added:** 18 (8 Task 1 + 10 Task 2 — all passing on first GREEN after one trivial test fix for alpha-aware color comparison)

## Accomplishments

- **All 7 Plottables in the catalog now ship.** Final inventory:
    1. `Vector` (polymorphic 2D / 3D per D-02) — 09-01
    2. `VectorComponents` (2D) — 09-01
    3. `Annotation` (2D with data + axes transforms) — 09-01
    4. `TimeSeries` (1D signal vs time) — 09-03
    5. `Heatmap` (2D imshow, log-freq y-tick option) — 09-03
    6. `Spotlight` (3 modes: rectangle / scatter / glow) — 09-03
    7. `Dropline` (dashed perpendicular) — 09-03

- **TimeSeries Plottable** at `research/dsplot/plottables/time_series.py` — `fill_between` of a 1D signal vs time. Default color resolves to `style.NEUTRAL_COLOR` because the palette reserves `PRIMARY`/`SECONDARY` for vector identity; a bare TimeSeries is non-identity data display. Background, tick label size, and tick color inherit from `dsplot.style` so a TimeSeries always renders consistently across figures.

- **Heatmap Plottable** at `research/dsplot/plottables/heatmap.py` — `imshow` of a 2D array. When `log_freq=True` and `freqs` is provided, y-tick labels are placed at `(20, 200, 2000, 20000)` Hz by default via `freq_axis.compute_freq_yticks`. vmax resolution has three-tier precedence: explicit `vmax` > `vmax_percentile` kwarg > `style.DEFAULT_HEATMAP_VMAX_PERCENTILE`. cmap defaults to `style.DEFAULT_HEATMAP_CMAP` via lazy lookup — a global reassignment between construction and draw is observable on the resulting image (pinned by Test 8).

- **Spotlight Plottable** at `research/dsplot/plottables/spotlight.py` — three rendering modes:
    - `"rectangle"`: `axvspan(x_range)` and/or `axhspan(y_range)`; both may be set, rendering two crossed spans.
    - `"scatter"`: single emphasized point at `xy` with white edge.
    - `"glow"`: alpha-graded `Circle` at `xy` with `radius`.
  Validation runs at `draw()` (not in `__init__`) — `mode="rectangle"` with neither range raises ValueError; an invalid `mode` raises ValueError listing the three valid modes. Default color is `style.TERTIARY_COLOR` (gold — the palette's high-visibility accent slot).

- **Dropline Plottable** at `research/dsplot/plottables/dropline.py` — dashed perpendicular between two points. All four style knobs (color / alpha / linewidth / linestyle) default to the `style.DROPLINE_*` and `style.DEFAULT_DROPLINE_*` constants and resolve lazily at `draw()` time. Reassigning `style.DROPLINE_COLOR` between construction and draw is observable on the rendered Line2D (pinned by Test 10).

- **freq_axis helper** at `research/dsplot/freq_axis.py` — `compute_freq_yticks(freqs, tick_freqs=(20, 200, 2000, 20000))` maps a sorted log-spaced frequency array to bin positions for a CWT-style heatmap y-axis. Labels format as `"{khz}k"` for ≥1000 Hz and `"{int(hz)}"` otherwise. Standalone (no project-specific imports) so it can drop into any consumer that wants log-frequency tick placement.

- **Top-level re-exports** wired so `from dsplot import TimeSeries, Heatmap, Spotlight, Dropline, freq_axis` all work. `__all__` extended to ten entries (style + freq_axis + 7 Plottables + Panel + StaticPanel + Figure).

## D-05 Lazy-Lookup Uniformity Check

All four new Plottables resolve None style knobs against `dsplot.style.*` INSIDE `draw()`, never at `__init__`:

| Plottable | Lazy-resolved knobs | Test that pins it |
|-----------|-------------|---|
| TimeSeries | `color` → `style.NEUTRAL_COLOR`; tick chrome direct | Test 4 (default-color via fill_between facecolor) |
| Heatmap | `cmap` → `style.DEFAULT_HEATMAP_CMAP`; `vmax_percentile` fallback → `style.DEFAULT_HEATMAP_VMAX_PERCENTILE` | **Test 8** (DEFAULT_HEATMAP_CMAP reassign-and-draw) |
| Spotlight | `color` → `style.TERTIARY_COLOR` | Implicit via artist-existence (no D-05 reassign test — added in a future plan if drift detected) |
| Dropline | `color` → `style.DROPLINE_COLOR`; `alpha` → `style.DEFAULT_DROPLINE_ALPHA`; `linewidth` → `style.DEFAULT_DROPLINE_LINEWIDTH`; `linestyle` → `style.DEFAULT_DROPLINE_LINESTYLE` | **Test 10** (DROPLINE_COLOR reassign-and-draw) |

Two dedicated D-05 tests (Heatmap Test 8 and Dropline Test 10) pin the lazy contract; the same pattern applies to TimeSeries and Spotlight by code review.

## Default-Color Rationale for Non-Identity Plottables

The dsplot palette assigns identity roles:
- `PRIMARY_COLOR` (orange) = "vector a identity"
- `SECONDARY_COLOR` (purple) = "vector b identity"

Plottables that DO NOT carry vector identity must NOT default to PRIMARY/SECONDARY. Their default-color choices:

| Plottable | Default color | Reason |
|-----------|--------------|--------|
| TimeSeries | `NEUTRAL_COLOR` | Generic data display — non-identity. If a figure wants a branded chirp (e.g. orange to match a downstream Vector), it passes `color=` explicitly. |
| Heatmap | (no `color`; uses `cmap` instead) | `cmap` controls the entire color scale; default `DEFAULT_HEATMAP_CMAP = "inferno"`. |
| Spotlight | `TERTIARY_COLOR` (gold) | High-visibility accent. Spotlight is foreground emphasis — needs to read above any data Plottable below it. |
| Dropline | `DROPLINE_COLOR` (gray) | Geometric construction line (perpendicular indicator). Dedicated palette slot, not data identity. |

Documented in the TimeSeries class docstring (longest justification; the others are unambiguous from their palette slot names).

## Task Commits

Each task followed the RED → GREEN TDD cycle.

1. **Task 1 RED** — `e1ed17f` — `test(09-03)`: failing tests for TimeSeries + Heatmap + freq_axis (8 behaviors; collection failed because of missing imports — RED gate)
2. **Task 1 GREEN** — `ecc0253` — `feat(09-03)`: implement TimeSeries + Heatmap + freq_axis (8 tests pass; full dsplot suite 79/79)
3. **Task 2 RED** — `851e2ef` — `test(09-03)`: failing tests for Spotlight + Dropline (10 behaviors; collection failed for missing imports — RED gate)
4. **Task 2 GREEN** — `821d19f` — `feat(09-03)`: implement Spotlight + Dropline (10 tests pass; full dsplot suite 89/89)

No REFACTOR commits were needed.

## Files Created / Modified

### Created — Library proper

- `research/dsplot/freq_axis.py` — `compute_freq_yticks` log-frequency tick helper
- `research/dsplot/plottables/time_series.py` — `TimeSeries` Plottable
- `research/dsplot/plottables/heatmap.py` — `Heatmap` Plottable
- `research/dsplot/plottables/spotlight.py` — `Spotlight` Plottable (3 modes)
- `research/dsplot/plottables/dropline.py` — `Dropline` Plottable

### Created — Tests

- `research/tests/dsplot/test_time_series_heatmap.py` — 8 tests covering compute_freq_yticks default + custom ticks, TimeSeries draw + default-color, Heatmap draw + log_freq y-ticks + vmax_percentile + D-05 lazy cmap lookup
- `research/tests/dsplot/test_spotlight_dropline.py` — 10 tests covering Spotlight 3 modes + 2 ValueError paths, Dropline default styling + explicit-color override + D-05 lazy DROPLINE_COLOR lookup

### Modified

- `research/dsplot/__init__.py` — `__all__` extended with `freq_axis`, `TimeSeries`, `Heatmap`, `Spotlight`, `Dropline`
- `research/dsplot/plottables/__init__.py` — re-exports for the four new Plottables

## Decisions Made

- **TimeSeries default color = NEUTRAL.** The palette reserves PRIMARY/SECONDARY for vector identity; a bare TimeSeries is non-identity data display. A figure that wants a branded chirp passes `color=` explicitly. Documented in the TimeSeries class docstring.
- **Heatmap vmax precedence (explicit > vmax_percentile > style default).** Test 7 pins it. Easy to invert; the test guards against regression.
- **Spotlight rectangle supports both x_range AND y_range.** Locked by Test 3. The alignment-diagnostic uses cyan vertical + horizontal markers; having both on one Spotlight is more ergonomic than composing two.
- **Spotlight validation defers to draw().** Matches the rest of the library — a misconfigured Plottable can be constructed and replaced before render without immediate failure.
- **Dropline uses `_raw_alpha` / `_raw_linestyle` shadow attributes** to preserve None-detection through the Plottable base class's concrete defaults. Considered relaxing the base class's defaults to None but rejected (would ripple into 5 existing Plottables for one consumer's need).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test 4 (TimeSeries default color) compared full RGBA instead of RGB**
- **Found during:** Task 1 GREEN — the test compared `to_rgba(NEUTRAL_COLOR)` against the fill_between facecolor, but `fill_between` applied `alpha=0.75` independently, so the facecolor's alpha was 0.75 — not 1.0 from `to_rgba`. The RGB channels did match.
- **Fix:** Compare RGB-only (`to_rgb` + slice [:3] on the actual rgba).
- **Files modified:** `research/tests/dsplot/test_time_series_heatmap.py`
- **Commit:** `ecc0253` (Task 1 GREEN — fix folded into same commit)
- **Why this is a test bug, not a code bug:** The plan's Behavior 4 says "TimeSeries default color resolves to `style.NEUTRAL_COLOR`" — that's about the COLOR (RGB), not the rendered RGBA. alpha is a separately-controlled style knob and is correctly 0.75 per the constructor default. The test had encoded the wrong fidelity check.

**2. [Rule 3 - Blocking] freq_axis.py docstring contained the literal word "subshader"**
- **Found during:** Task 1 GREEN regression run — `test_library_has_zero_subshader_imports` (09-01's D-01 isolation guard) failed because my docstring said "Standalone (no subshader imports)..." — a substring match.
- **Fix:** Rewrote the docstring as "Pure numpy helper — no external project imports." Same meaning, no trigger word.
- **Files modified:** `research/dsplot/freq_axis.py`
- **Commit:** `ecc0253` (Task 1 GREEN — fix folded into same commit)
- **Why I did NOT weaken the test:** The test scans for substring "subshader" in `.py` files inside `research/dsplot/` (excluding `figures/`). Weakening it to a stricter regex would risk masking actual `import subshader` lines that someone could add later. The test's bluntness is intentional — D-01 wants zero mentions, period. The fix is on the offender side.

### No other deviations

No checkpoints hit. No Rule 4 architectural decisions needed. No CLAUDE.md violations (descriptive names; no comment litter; no new dependencies). No auth gates.

## Issues Encountered

- **Worktree path discipline.** When invoking commands without explicit `cd`, the shell defaults to the worktree root at `.claude/worktrees/agent-a600bb530e0513ae8/`. An early `cd /home/eddie-water/dev/python/sub-shader && ...` accidentally operated on the MAIN repo (which is on the `gsd/phase-09-...` branch, not the per-agent worktree branch). The HEAD-namespace assertion in the executor protocol caught it immediately (commit refused on the protected branch). Resolution: moved the misplaced test file from the main-repo `research/tests/dsplot/` to the worktree's matching path, unstaged the main-repo `research/tests/dsplot/test_time_series_heatmap.py`, and ran all subsequent Write/Edit/Bash via the worktree-absolute path. No commits landed in the main repo — the index was reset before commit.
- No other issues. Both tasks passed GREEN on first run after the Task 1 test fix and the docstring fix described above.

## Out-of-Scope Items Encountered

- The shared parent-repo working tree contains a large set of untracked files (palette schemas, generated PNGs, planning artifacts from other agents). None of these were touched. The worktree was reset to its base (`89e6eed`) at agent start, so this run sees only the 09-01 and 09-02 outputs plus the new 09-03 artifacts.
- The 09-02 SUMMARY notes that `style.DEFAULT_AXIS_GRID_*` names trip a strict reading of non-goal #5 ("no GRID_* constants"). This is unchanged by 09-03 — I did not rename them. If a future plan decides to rename, it's a one-line-per-constant change. Logged here only for visibility.

## User Setup Required

None.

## Next Phase Readiness

- **Wave 3 figure ports** can now compose every dispatched figure in `research/dsp_figures.py` from the public dsplot API:
    - `motivator` chirps → `TimeSeries` (audio panel) + `Heatmap` (CWT panel) on a `Figure` grid.
    - `alignment_diagnostic` → `TimeSeries` + `Heatmap` + `Spotlight(mode="rectangle", x_range=...)` for the burst-time vertical band + `Spotlight(..., y_range=...)` for each horizontal burst-freq marker, all stacked on a 2-row `Figure`.
    - `vector_basics`, `components_recombine_either_order`, `projection_reconstruction_either_order`, `dot_product_geometry` → `Vector` + `VectorComponents` + `Annotation` + `Dropline` on `StaticPanel`s.
    - `vector_projection_3d` → polymorphic `Vector` with 3-tuple inputs on a `Figure.add_panel(..., projection="3d")` axes (D-04 + D-02 lock — no Vector3D class).
- **`compute_freq_yticks` is reusable** — any future log-frequency panel (FFT spectrum, STFT spectrogram, instantaneous-frequency curve) can use it directly without re-implementing the bin interpolation + label formatting.
- **D-05 lazy-lookup contract is now exercised by 4 separate Plottables** (Vector, Annotation, Heatmap, Dropline) with explicit reassign-and-draw tests. Future Plottables should follow the same pattern; the 4 examples are a sufficient template.

## Self-Check: PASSED

Created files verified to exist (worktree paths):
- FOUND: research/dsplot/freq_axis.py
- FOUND: research/dsplot/plottables/time_series.py
- FOUND: research/dsplot/plottables/heatmap.py
- FOUND: research/dsplot/plottables/spotlight.py
- FOUND: research/dsplot/plottables/dropline.py
- FOUND: research/tests/dsplot/test_time_series_heatmap.py
- FOUND: research/tests/dsplot/test_spotlight_dropline.py

Commits verified to exist in git log:
- FOUND: e1ed17f (test, Task 1 RED)
- FOUND: ecc0253 (feat, Task 1 GREEN)
- FOUND: 851e2ef (test, Task 2 RED)
- FOUND: 821d19f (feat, Task 2 GREEN)

Verification commands all pass:
- `python -m pytest research/tests/dsplot/ -q` → 89 passed (71 baseline + 8 + 10)
- `cd research && python -c "from dsplot import Vector, VectorComponents, Annotation, TimeSeries, Heatmap, Spotlight, Dropline, Panel, StaticPanel, Figure, freq_axis; Vector((1,2,3))"` → exits 0
- `grep -rIn "subshader" research/dsplot/ --exclude-dir=figures` → zero matches (D-01 LOCKED)
- `grep -rIn "Vector3D" research/dsplot/` → zero matches (D-02 LOCKED — 3D rendering is on polymorphic Vector)

## TDD Gate Compliance

Both tasks followed the RED → GREEN cycle:

- Task 1: `test(...)` commit `e1ed17f` precedes `feat(...)` commit `ecc0253`
- Task 2: `test(...)` commit `851e2ef` precedes `feat(...)` commit `821d19f`

Each RED commit was verified to fail (ImportError on missing exports), establishing the actual fail signal before the matching GREEN landed.

---
*Phase: 09-dsplot-library-architecture*
*Plan: 03*
*Completed: 2026-05-17*
