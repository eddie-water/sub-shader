---
phase: 09-dsplot-library-architecture
plan: 01
subsystem: plotting

tags: [dsplot, matplotlib, plotting-library, plottable, vector, annotation, style-template]

# Dependency graph
requires:
  - phase: 07-visual-style-system-and-frequency-range-configuration
    provides: research/utilities/style.py — source palette/typography defaults; dsplot.style generalizes them
  - phase: 08-codebase-refactoring-and-module-cleanup
    provides: research/dsp_figures.py monolith — figures Layer that 09 decomposes into Plottables + Panels
provides:
  - dsplot package skeleton at research/dsplot/
  - dsplot.style inheritable template (palette + typography + linework + arrowheads + layout + heatmap defaults)
  - Plottable abstract base class with .draw(ax) contract and lazy style lookup
  - Vector Plottable (polymorphic 2D/3D per D-02, asymmetric dispatch per D-06)
  - VectorComponents Plottable (2D only — x/y decomposition with droplines)
  - Annotation Plottable (2D, with transform="axes" axes-relative placement)
affects: [09-02-panels, 09-03-time-series-heatmap, 09-04-3d-foundation, 09-05-figure-ports, 09-06-notebook-demo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy style-lookup: Plottable.__init__ stores None for unset knobs; draw() resolves None against dsplot.style.* at render time — runtime reassignment between construct and draw is observable"
    - "Polymorphic Plottable dispatch on input shape: Vector accepts 2-tuple OR 3-tuple, branches on len(self.vec) inside draw()"
    - "Asymmetric dimensional dispatch (D-06): 3-tuple onto 2D Axes → TypeError (no defensible 3D→2D collapse); 2-tuple onto Axes3D → silent z=0 extension"
    - "Dashed-arrow split (2D Vector): ax.plot for the dashed shaft + solid FancyArrowPatch stub for just the head, avoiding the matplotlib dashed-arrowhead ghosting"

key-files:
  created:
    - research/dsplot/__init__.py
    - research/dsplot/style.py
    - research/dsplot/plottables/__init__.py
    - research/dsplot/plottables/base.py
    - research/dsplot/plottables/vector.py
    - research/dsplot/plottables/vector_components.py
    - research/dsplot/plottables/annotation.py
    - research/tests/dsplot/__init__.py
    - research/tests/dsplot/test_plottable_construction.py
    - research/tests/dsplot/test_vector_plottables.py
    - research/tests/dsplot/test_annotation.py
    - .planning/phases/09-dsplot-library-architecture/deferred-items.md
  modified:
    - pyproject.toml

key-decisions:
  - "pyproject.toml pytest pythonpath now includes research/ so dsplot is importable from pytest without requiring tests to be in the same package — needed because dsplot lives at research/dsplot/ but tests live at research/tests/dsplot/"
  - "VectorComponents delegates to Vector for each component arrow (composition over duplication) — keeps the dashed-shaft + solid-head workaround in one place"
  - "Annotation.draw() routes arrow-callout requests through ax.annotate (with arrowprops) and plain-text requests through ax.text (with optional transform=ax.transAxes) — the two paths produce different mpl artist trees, kept separate intentionally"
  - "Default 3D color is NEUTRAL_COLOR (off-white) per the existing _plot_vector_projection_3d convention — 2D defaults to PRIMARY_COLOR (orange) per the 2D vector-a convention"

patterns-established:
  - "lazy-style-lookup: None at construction, style.* at draw"
  - "polymorphic-Plottable-by-tuple-length: D-02"
  - "asymmetric-dimensional-dispatch: D-06 (2D→3D implicit z=0; 3D→2D loud TypeError)"
  - "dashed-arrow-split: shaft + solid-head stub"
  - "delegated-composition: VectorComponents builds on Vector"

requirements-completed: [PLOT-01, PLOT-02, PLOT-03, PLOT-04]

# Metrics
duration: 9min
completed: 2026-05-17
---

# Phase 09 Plan 01: dsplot Library Foundation Summary

**dsplot package skeleton with inheritable style template (D-05), Plottable abstract base, and three concrete Plottables (Vector polymorphic 2D/3D per D-02, VectorComponents 2D, Annotation 2D with axes-relative transform) — zero subshader imports.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-17T17:41:56Z
- **Completed:** 2026-05-17T17:50:05Z
- **Tasks:** 3 (all TDD)
- **Files created:** 12
- **Files modified:** 1
- **Tests added:** 53 (all passing)

## Accomplishments

- **dsplot package importable** at `research/dsplot/` with zero `subshader` imports (D-01 isolation grep returns 0 matches with `--exclude-dir=figures`).
- **Inheritable style template (D-05)** in `dsplot/style.py` — comprehensive coverage: role-named palette (PRIMARY/SECONDARY/TERTIARY/NEUTRAL/HIGHLIGHT/BG/SPINE/TICK_LABEL/DROPLINE) + `DEFAULT_*` defaults for typography (title/subtitle/label/tick/suptitle/row-label), linework (vector/spine/dropline widths and alphas), arrowheads (head length/width/mutation), panel sizing (DPI, panel size, margins), row/grid layout (HSPACE/WSPACE/LABEL_RATIO), vector-axes decoration, and heatmap defaults.
- **Plottable abstract base** at `plottables/base.py` — defines the `.draw(ax)` contract with shared style knobs (color, linewidth, alpha, linestyle, label, zorder). None-valued knobs are resolved at `draw()` time against `dsplot.style.*` (lazy lookup — what makes runtime style reassignment work, per D-05).
- **Polymorphic Vector (D-02)** — a single `Vector` class accepts a 2-tuple or 3-tuple. Branches at `draw()`: 2-tuple → 2D FancyArrowPatch shaft + head; 3-tuple → 3D `ax.plot` + scatter tip + 3D text. No separate `Vector3D` class. Asymmetric dispatch per D-06: 3-tuple on a 2D Axes raises `TypeError`, 2-tuple on `Axes3D` silently extends to `(x, y, 0)`.
- **2D dashed-arrow ghosting workaround preserved** — when `linestyle != "-"` the 2D branch splits the arrow into `ax.plot` for the dashed shaft + a solid `FancyArrowPatch` stub (1e-3 along the heading direction) for just the head, so the dash pattern never reaches the arrowhead outline.
- **VectorComponents (2D only)** — x/y decomposition; delegates to `Vector` for each component arrow; optional droplines from the tip to each axis; reorderable via `first_axis="x"|"y"`.
- **Annotation (2D)** — text at `(x, y)` with optional arrow callout; `transform="data"` (default) or `transform="axes"` (axes-relative, supports the "result text below panel" pattern used by 09-05's `dot_product_geometry`).
- **Top-level re-exports** wired so `from dsplot import Vector, VectorComponents, Annotation` works.

## Task Commits

Each task followed RED/GREEN TDD cycle.

1. **Task 1 RED — style template + Plottable base tests** — `a78dc3d` (test)
2. **Task 1 GREEN — dsplot package skeleton + style.py + Plottable base** — `690d68a` (feat)
3. **Task 1 — deferred-items.md (orphan test from sibling 09-02)** — `1d5f73a` (docs)
4. **Task 2 RED — Vector + VectorComponents tests** — `569b776` (test)
5. **Task 2 GREEN — Vector (polymorphic 2D/3D) + VectorComponents** — `b1250dd` (feat) [subject mislabeled — see Deviations]
6. **Task 3 RED — Annotation tests** — `38f4bfe` (test)
7. **Task 3 GREEN — Annotation + top-level re-exports** — `26e4f29` (feat)

## Files Created/Modified

### Created — Library proper
- `research/dsplot/__init__.py` — top-level package; re-exports `style`, `Vector`, `VectorComponents`, `Annotation`
- `research/dsplot/style.py` — D-05 inheritable template (palette + DEFAULT_* layout/typography/linework/arrowhead/heatmap constants)
- `research/dsplot/plottables/__init__.py` — re-exports the concrete Plottables
- `research/dsplot/plottables/base.py` — `Plottable` abstract base with `.draw(ax)` contract
- `research/dsplot/plottables/vector.py` — `Vector` polymorphic on tuple length (D-02)
- `research/dsplot/plottables/vector_components.py` — `VectorComponents` (2D only)
- `research/dsplot/plottables/annotation.py` — `Annotation` (2D + `transform="axes"`)

### Created — Tests
- `research/tests/dsplot/__init__.py`
- `research/tests/dsplot/test_plottable_construction.py` — 30 tests (D-01 isolation, D-05 global/local override, template completeness, Plottable contract)
- `research/tests/dsplot/test_vector_plottables.py` — 17 tests (D-02 polymorphic 2D/3D, D-06 asymmetric dispatch, D-05 lazy lookup, dashed-arrow split, VectorComponents)
- `research/tests/dsplot/test_annotation.py` — 6 tests (data-coord placement, axes-relative transform, arrow callouts, D-05 lazy lookup, top-level imports)

### Created — Planning
- `.planning/phases/09-dsplot-library-architecture/deferred-items.md` — logs the orphan `test_panel_composition.py` belonging to sibling plan 09-02 (already untracked on the worktree at 09-01 start)

### Modified
- `pyproject.toml` — added `research` to pytest `pythonpath` so `import dsplot` works from `research/tests/dsplot/*.py` (otherwise `dsplot` is only resolvable when pytest's cwd happens to be `research/`)

## Decisions Made

- **pyproject.toml pytest pythonpath addition** — research/ was added to pytest pythonpath as the cleanest way to make `import dsplot` succeed under `python -m pytest` regardless of cwd. Alternative considered: a conftest.py that inserts `research/` into `sys.path` at test-collection time. The pyproject change is more discoverable and lives next to the other pythonpath entries.
- **VectorComponents delegates to Vector** rather than redrawing arrows itself — keeps the dashed-shaft + solid-head workaround in exactly one place (`Vector._draw_2d`). When future work tunes head dimensions or alpha, VectorComponents inherits the fix automatically.
- **3D default color = NEUTRAL_COLOR** — matches the existing `_plot_vector_projection_3d` convention where vector `a` renders as off-white because the 3D scene's color drama is paid out by the per-segment dimensional path coloring (x=primary, y=secondary, z=tertiary), not by vector identity.
- **D-06 implementation as a hasattr check** — the 2D branch uses `if _is_axes_3d(ax): return self._draw_3d(...)` to dispatch a 2-tuple onto Axes3D; `_is_axes_3d` is just `hasattr(ax, "get_zlim")`. Avoids the import dependency that `isinstance(ax, Axes3D)` would create.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `research` to pytest `pythonpath` in pyproject.toml**
- **Found during:** Task 1 (RED test setup)
- **Issue:** Tests at `research/tests/dsplot/*.py` cannot `import dsplot` because pyproject's `pythonpath` was `["src", "src/subshader", "research/tests"]` — `research/` itself was missing, so `dsplot` (which lives at `research/dsplot/`) was not resolvable when pytest runs.
- **Fix:** Inserted `"research"` into the pytest `pythonpath` list. Other research/ tests (`tests/dsp/test_wavelet.py` etc.) continue to work because their imports are already qualified.
- **Files modified:** `pyproject.toml`
- **Verification:** `cd research && python -m pytest tests/dsplot/ -q` collects and runs 53 tests successfully.
- **Committed in:** `a78dc3d` (Task 1 RED)

### Process anomalies (not deviations from plan code)

**Commit subject mismatch on task 2 GREEN (`b1250dd`)**
- The Task 2 GREEN commit's subject line reads `test(09-02): add failing tests for Panel ABC + StaticPanel + axes_setup helper` but the staged diff is in fact the Task 2 GREEN code from 09-01 (Vector + VectorComponents implementation). The intended subject was `feat(09-01): implement Vector (polymorphic 2D/3D per D-02) + VectorComponents (2D)`.
- The body of `.git/COMMIT_EDITMSG` confirms the executor's intended message was prepared correctly; the commit object on HEAD instead carries a 09-02-flavoured subject and body. Working theory: another parallel-wave executor (running plan 09-02 in this same repository — see "Orphan files" below) interleaved a `git commit` between this executor's `git add` and `git commit`, and the index it staged ended up captured under the other executor's message. Either way, the committed artifact contents match this plan's Task 2 GREEN exactly (vector.py, vector_components.py, plottables/__init__.py, dsplot/__init__.py).
- Per the project's commit-protocol guard ("CRITICAL: Always create NEW commits rather than amending") and `<destructive_git_prohibition>`, the commit was NOT amended. The subject mismatch is cosmetic — the artifact and the test suite both reflect the intended Task 2 GREEN.

### Orphan files left in place (out-of-scope per scope-boundary rule)

The worktree contained the following untracked files at the start of 09-01 execution — they belong to sibling plan 09-02 and were left as-is. Logged in `.planning/phases/09-dsplot-library-architecture/deferred-items.md`.

- `research/tests/dsplot/test_panel_composition.py` (09-02 Panel composition tests)
- `research/dsplot/axes_setup.py` (09-02 axes setup helper)
- `research/dsplot/panels/` (09-02 Panel ABC + StaticPanel)

These are not in any 09-01 commit and remain untracked in the worktree.

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking import path); 1 process anomaly (commit subject mismatch on `b1250dd`).
**Impact on plan:** Auto-fix was necessary for tests to run at all. Commit subject mismatch is cosmetic — artifact contents and test coverage match the plan.

## Issues Encountered

- **`grep -rIn` returns exit 1 when there are zero matches.** This is correct grep behavior ("no lines matched") but reads like a failure in shell pipelines. Worked around by writing the grep output to a file and testing `if [ -s file ]`.
- **Multiple plans share `research/dsplot/` directory.** Plan 09-02 had already deposited orphan files (`axes_setup.py`, `panels/__init__.py`, `panels/base.py`, `panels/static_panel.py`, `tests/dsplot/test_panel_composition.py`) on disk before this executor started. Resolution: leave them untouched (destructive-git prohibition), unstage anything accidentally picked up by `git add`, log to `deferred-items.md` for the orchestrator. The 09-02 executor's work is intact.

## User Setup Required

None.

## Next Phase Readiness

- **dsplot Plottable contract is locked in:** every later plan implements `.draw(ax)` against an Axes and resolves None-valued style knobs against `dsplot.style.*` at draw time.
- **Inheritable-template rule for future plans:** if a figure-port plan discovers it needs a layout constant not yet in `dsplot.style`, ADD it as a `DEFAULT_*` constant in `dsplot/style.py` and either consume it directly or derive a figure-local override. NEVER hardcode the value at the figure call site.
- **09-02 Panels** can build on the existing `Plottable` contract with confidence — the contract is exercised by 53 tests.
- **09-03 TimeSeries / Heatmap / Spotlight / Dropline** can mirror the lazy-lookup pattern from `Vector` / `VectorComponents` / `Annotation`.
- **09-04 3D foundation figure** has the polymorphic Vector already 3D-capable; the figure port mostly needs scene setup (manual spines, view_init) which 09-04 owns.
- **09-05 figure ports** can use `Annotation(text, xy, transform="axes")` for the result-text-below-panel pattern.

## Self-Check: PASSED

Created files verified to exist:
- FOUND: research/dsplot/__init__.py
- FOUND: research/dsplot/style.py
- FOUND: research/dsplot/plottables/__init__.py
- FOUND: research/dsplot/plottables/base.py
- FOUND: research/dsplot/plottables/vector.py
- FOUND: research/dsplot/plottables/vector_components.py
- FOUND: research/dsplot/plottables/annotation.py
- FOUND: research/tests/dsplot/__init__.py
- FOUND: research/tests/dsplot/test_plottable_construction.py
- FOUND: research/tests/dsplot/test_vector_plottables.py
- FOUND: research/tests/dsplot/test_annotation.py
- FOUND: .planning/phases/09-dsplot-library-architecture/deferred-items.md

Commits verified to exist:
- FOUND: a78dc3d (test, Task 1 RED)
- FOUND: 690d68a (feat, Task 1 GREEN)
- FOUND: 1d5f73a (docs, deferred-items)
- FOUND: 569b776 (test, Task 2 RED)
- FOUND: b1250dd (feat — subject mislabeled but artifact matches Task 2 GREEN)
- FOUND: 38f4bfe (test, Task 3 RED)
- FOUND: 26e4f29 (feat, Task 3 GREEN)

Verification commands:
- `cd research && python -c "import dsplot; import dsplot.style; from dsplot import Vector, VectorComponents, Annotation"` → exits 0 [PASS]
- `grep -rIn "subshader" research/dsplot/ --exclude-dir=figures` → 0 matches [PASS — D-01]
- `cd research && python -m pytest tests/dsplot/test_plottable_construction.py tests/dsplot/test_vector_plottables.py tests/dsplot/test_annotation.py -q` → 53 passed [PASS]

## TDD Gate Compliance

All three tasks followed the RED→GREEN cycle:
- Task 1: test commit `a78dc3d` precedes feat commit `690d68a`
- Task 2: test commit `569b776` precedes feat commit `b1250dd`
- Task 3: test commit `38f4bfe` precedes feat commit `26e4f29`

No REFACTOR commits were needed — implementations were minimal and passed first GREEN run.

---
*Phase: 09-dsplot-library-architecture*
*Plan: 01*
*Completed: 2026-05-17*
