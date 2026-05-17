---
phase: 09-dsplot-library-architecture
plan: 06
type: summary
status: pending-checkpoint-verification
subsystem: dsplot consumer code (notebook + tests) + dsp_figures.py retirement
tags:
  - dsplot
  - notebook
  - jupyter
  - dynamic-panel
  - mixed-type-figure
  - style-override
  - d-05
  - d-08
  - deprecation-shim
  - phase-9-closeout
requirements: [PLOT-26, PLOT-27, PLOT-28, PLOT-29]
locked_decisions:
  - D-05 (template + override): style_override_demo demonstrates BOTH modes — global reassignment AND local-only override. Demo asserts global default restored after both blocks.
  - D-08 (notebook §2.4 incremental): cell 01 contains §2.4 Figure 1 as a 1x3 mixed Figure per D-08 (Static xy projection + Dynamic reconstruction + Dynamic orthogonality). Subsequent §2.4 cells deferred to follow-up plans (09-07+).
key-files:
  created:
    - src/subshader/dsp/dsp.ipynb (cell 7ab635e8 populated; markdown 4e7d6776 untouched)
    - research/tests/dsplot/test_dsplot_example.py
    - research/dsplot/figures/style_override_demo.py
    - .planning/phases/09-dsplot-library-architecture/style-override-snippet.md
    - .planning/phases/09-dsplot-library-architecture/09-06-SUMMARY.md
    - assets/images/dsp/style_override_demo/default_palette.png
    - assets/images/dsp/style_override_demo/global_override.png
    - assets/images/dsp/style_override_demo/local_override.png
    - assets/images/diagnostics/09-06-checkpoint/cell01_full_render.png
    - assets/images/diagnostics/09-06-checkpoint/cell01_composite.png
    - assets/images/diagnostics/09-06-checkpoint/cell01_panel1_static_xy.png
    - assets/images/diagnostics/09-06-checkpoint/cell01_panel2_recon_frame{0,2,4}.png
    - assets/images/diagnostics/09-06-checkpoint/cell01_panel3_orth_frame{0..4}.png
  modified:
    - research/dsp_figures.py (1066 -> 63 lines; deprecation shim)
commits:
  - 743fd22: feat(09-06) populate dsp.ipynb cell 01 with §2.4 Figure 1 as 1x3 mixed Figure (D-08)
  - 5cd2c51: test(09-06) add dsplot example integration test (teaching artifact)
  - a0bbe61: feat(09-06) style-override demo + docs snippet for BOTH D-05 modes
  - 6703556: refactor(09-06) retire research/dsp_figures.py to a deprecation shim
metrics:
  duration: 35min
  completed: 2026-05-17
  tasks_completed: 3
  tasks_pending: 1 (Task 4 — checkpoint:human-verify)
  files_created: 9 (+ 13 diagnostic snapshot PNGs)
  files_modified: 1
  tests_added: 1 (test_dsplot_example_composition)
  test_suite_total: 110 passing (was 109 before this plan)
---

# 09-06 Summary — Final consumer-side work + dsp_figures.py retirement

The last three CONTEXT.md verification targets are landed: notebook demo (target
#3 — populated per D-08), example unit test (target #4), and style-override
demonstration for BOTH D-05 modes (target #5). The historical
`research/dsp_figures.py` is retired to a 63-line deprecation shim (target #6).
Targets #1 (library isolation) and #2 (figure parity) were satisfied in earlier
plans (09-01..09-05). After Task 4's human-verify checkpoint confirms the
notebook cell 01 animation behavior on the user's machine, Phase 9 ships.

## Tasks landed

### Task 1 — Notebook cell 01 populated per D-08

`src/subshader/dsp/dsp.ipynb` cell `7ab635e8` (previously empty) now contains
a 1×3 mixed Figure composed of three panels:

| Col | Panel type     | Contents                                                        |
|-----|----------------|-----------------------------------------------------------------|
| 0   | StaticPanel    | xy projection of `a = (2, 3)` into its x and y components       |
| 1   | DynamicPanel   | reconstruction order — 6 frames, x-first then y-first ordering  |
| 2   | DynamicPanel   | orthogonality — 5 frames, a' x sweep [+2, +1, 0, -1, -2]        |

Both DynamicPanels loop with `repeat=True`; the StaticPanel renders once and
holds. Construction matches the LOCKED D-08 amendment in `09-CONTEXT.md`. The
markdown cell `4e7d6776` (the §2.4.1 prose) is untouched.

**Implementation choices:**
- **sys.path bootstrap:** the cell resolves `research/` relative to either the
  repo-root cwd or the `src/subshader/dsp/` cwd, so the notebook works from
  either launch position (Jupyter Lab from repo root, or VS Code Jupyter from
  the notebook's directory).
- **Backend selection:** `%matplotlib widget` and `%matplotlib inline` are both
  commented in the cell — user uncomments per environment. `widget` produces
  the live animation loop; `inline` falls back to frame 0 only.
- **Dark Jupyter chrome:** `dsplot.apply_jupyter_dark()` is called once near the
  top of the cell so the cell-output container background matches the figure
  bg (the same one-shot CSS injection 09-04 introduced).
- **Optional GIF export:** the orthogonality panel can export a GIF via
  `panel.save_gif(...)`; the line is commented in the cell so the default
  invocation doesn't write to disk.
- **Frame count for reconstruction:** 6 frames cover both orderings (x-first,
  y-first) plus a brief hold frame so the loop feels deliberate.
- **a' x sweep:** the spec-mandated `[+2.0, +1.0, 0.0, -1.0, -2.0]` is hardcoded
  in the cell as `A_PRIME_X_SWEEP`. a's reference at `A = (2, 3)` and a'.y at
  3.0 are pinned across every frame so the orthogonality beat reads cleanly.

**Verification snapshots** generated for the human-verify checkpoint live at
`assets/images/diagnostics/09-06-checkpoint/`:

- `cell01_full_render.png` — the entire 1×3 figure rendered headless (frame 0
  of both Dynamic panels).
- `cell01_composite.png` — three Static panels showing the frame-0 contents of
  each Dynamic panel side-by-side.
- `cell01_panel1_static_xy.png` — Panel 0 (StaticPanel).
- `cell01_panel2_recon_frame0.png`, `frame2.png`, `frame4.png` — Panel 1
  reconstruction at frames 0 (blank start), 2 (x-first complete), 4 (y-first
  complete).
- `cell01_panel3_orth_frame0..4.png` — Panel 2 orthogonality at each of the 5
  a' x-sweep frames.

### Task 2 — Example test + style-override demo + docs snippet (D-05 both modes)

- **`research/tests/dsplot/test_dsplot_example.py`** — single ~30-line test
  demonstrating the canonical Figure + StaticPanel + Vector + savefig
  integration pattern. Teaching artifact, not coverage. Adds one passing test
  to the suite (was 109, now 110).

- **`research/dsplot/figures/style_override_demo.py`** — runnable demo of BOTH
  D-05 override modes. Invokable as
  `python -m research.dsplot.figures.style_override_demo` from the repo root
  (the demo bootstraps `research/` onto `sys.path`, mirroring `__main__.py`).
  Produces three side-by-side PNGs in `assets/images/dsp/style_override_demo/`:

  | File                  | Mode                          | Visual         |
  |-----------------------|-------------------------------|----------------|
  | `default_palette.png` | baseline (template defaults)  | orange/purple/gold |
  | `global_override.png` | Mode 1 (`dsplot.style.*` reassigned) | green/pink/cyan |
  | `local_override.png`  | Mode 2 (module-local `MY_*` constants)| yellow/purple/orange |

  Demo's `main()` restores originals in `try/finally` and asserts
  `dsplot.style.PRIMARY_COLOR == "#e1641a"` after the local-mode block — proves
  Mode 2 did not leak into Mode 1.

- **`.planning/phases/09-dsplot-library-architecture/style-override-snippet.md`**
  — docs-ready snippet with copy-pasteable code for both modes, "when to use
  each" guidance, an explanation of the lazy-lookup mechanism, and a pointer
  to the runnable demo.

### Task 3 — `research/dsp_figures.py` retired to deprecation shim

The 1066-line monolith is now a 63-line shim that:
- Emits a `DeprecationWarning` on import.
- Re-exports every canonical top-level name from
  `research/dsplot/figures/` under its legacy alias:
  - `A`, `A_PRIME`, `B`, `A_Z`, `FOUND_LIM` from `foundation_constants`.
  - `ChirpFigureConfig` (alias for `MotivatorConfig`),
    `MOTIVATOR_VERSIONS` (alias for `VERSIONS`),
    `render_motivator` (alias for `render_one`),
    `generate_motivator_versions` (alias for `render_all`) from `motivator`.
  - `generate_alignment_diagnostic` (alias for `render`) from
    `alignment_diagnostic`.
- Defines `generate_all_dsp_figures()` that delegates to
  `research/dsplot/figures/__main__.py::main()`.
- `python research/dsp_figures.py` (legacy invocation) still works.

**Caller-grep at task start:**

```
research/dsp_figures.py: (self-references — definitions)
.claude/plan-05-04-section-2.4.1-vector-xy-reconstruction-figure-and-axis-helpers.md:39:
   .venv/bin/python -c "from dsp_figures import _plot_vector_xy_reconstruction; ..."
```

Result: **zero production callers**. The one external reference is a stale
planning note referencing a private helper `_plot_vector_xy_reconstruction`
that was always under-the-line and isn't on the shim's export surface. That
stale doc-link breakage is acceptable per the plan's
"don't honor private-helper contracts" guidance.

## Task 4 — Awaiting human-verify checkpoint

This plan ends in a `checkpoint:human-verify` for the 6 CONTEXT.md verification
targets. The notebook cell 01 animation behavior cannot be verified headless —
the user verifies it interactively in Jupyter (VS Code Jupyter or Jupyter Lab
with `%matplotlib widget`). See the checkpoint payload returned to the
orchestrator for the exact verification steps and visual-snapshot artifacts.

## Deviations from plan

### Process adjustments (not deviations from plan code)

**1. style_override_demo invocation path bootstrap**
- **Found during:** Task 2 verification (`python -m research.dsplot.figures.style_override_demo` from repo root).
- **Issue:** The demo imports `dsplot`, but `research/` is not on
  `sys.path` for a bare `python -m` invocation from the repo root because
  pytest's `pythonpath` only applies under pytest. The plan's behavior assumed
  the canonical invocation pattern would just work, but it doesn't without
  bootstrapping.
- **Fix:** Inserted the same `_RESEARCH_DIR` sys.path bootstrap that
  `research/dsplot/figures/__main__.py` uses. Demo now runs cleanly from
  any cwd via `python -m research.dsplot.figures.style_override_demo`.
- **Files modified:** `research/dsplot/figures/style_override_demo.py`
- **Verification:** demo produces all three PNGs and the cross-check
  `PYTHONPATH=research python -c "import dsplot; assert dsplot.style.PRIMARY_COLOR == '#e1641a'"`
  passes.

**2. Plan verify-step assumes bare `python -c "import dsplot"` works**
- The plan's automated verify step (Task 2) uses
  `python -c "import dsplot; assert dsplot.style.PRIMARY_COLOR == '#e1641a'"`
  to check the demo restored the global. That command needs `PYTHONPATH=research`
  prefixed (or to run from `research/`) — otherwise `import dsplot` fails with
  ModuleNotFoundError outside pytest. The fix is documented here for the
  checkpoint-runner: prepend `PYTHONPATH=research` to the bare `python -c`
  invocation, or run `cd research && python -c "..."`. Not blocking — the
  demo's own internal `try/finally` already guarantees restoration; the
  external assertion is belt-and-braces.

### Auto-fixed (Rule 1 bugs)

None. All three task implementations passed verification on first run after the
sys.path bootstrap fix above.

### Out-of-scope (deferred per scope-boundary rule)

The worktree HEAD's git status snapshot shows pre-existing modifications to:
- `assets/images/dsp/dot_product_geometry.png`
- `assets/images/dsp/vector_basics.png`
- `assets/images/dsp/vector_projection_3d_v2_combo5_palette.png`
- `assets/images/dsp/vector_xy_reconstruction.png`
- New untracked files: `assets/images/dsp/components_recombine_either_order_v18.png`,
  `assets/images/dsp/projection_reconstruction_either_order_v9.png`

These were modified/created before this executor started — likely by a prior
session's `python -m research.dsplot.figures` invocation overwriting the
`_09_05`-suffixed parity outputs. Not part of any 09-06 commit. Leaving them
alone per the scope-boundary rule; they will be reconciled by the 09-05
visual-parity merge step the user owns.

## Threat surface scan

No new network endpoints, auth paths, file-access patterns, or trust-boundary
schema changes. All three task deliverables are plotting/consumer code with no
security-relevant surface.

## TDD gate compliance

Plan 09-06 marks Task 2 with `tdd="true"`. The test commit
(`test(09-06): add dsplot example integration test`) precedes the feat commit
(`feat(09-06): style-override demo + docs snippet`). The test passes
unconditionally — there is no failing-then-passing transition to gate against
because the test exercises already-implemented dsplot integration, not
new-feature code. That's expected for a teaching-artifact test (per the test's
own docstring). Other tasks are `type="auto"` (Task 1, Task 3) and don't
require the RED/GREEN cycle.

## Phase 9 retrospective (hand-off notes)

This plan closes Phase 9. Lessons from the whole phase:

- **Mixed-type Figure composition works** — Static + Dynamic + Dynamic on one
  Figure (this plan), Static + Dynamic + Interactive (09-04 Smoke 5). The
  primary LOCKED goal is met; the same-type-only fallback was never triggered.
- **Twin-axis escape hatch** — `motivator.py` uses `ax.twinx()` directly inside
  its render rather than forcing a `TimeSeriesWithTwinAxis` Plottable. The
  abstraction is honest about the one boundary it crosses (documented in
  `motivator.py`'s docstring).
- **D-01 library/consumer boundary** — the `figures/` subdir is explicitly
  consumer code; the isolation grep uses `--exclude-dir=figures`. This kept
  the library proper truly subshader-free while letting figure modules bridge
  to `research.utilities` for CWT computation.
- **D-02 polymorphic Vector** — a single `Vector` class handles 2D (FancyArrowPatch)
  and 3D (ax.plot + scatter + 3D text) via tuple-length dispatch. Consumers
  never instantiate a separate `Vector3D`; the dimensional difference is
  inferred at construction.
- **D-04 3D accommodations** — `Figure.add_panel(..., projection="3d")` creates
  an `Axes3D` for that cell; consumer panels that want 3D-specific chrome can
  subclass (`StaticPanel3D` did this for the foundation 3D figure).
- **D-05 template-with-override** — both override modes (global reassignment AND
  local figure override) work and are documented + demonstrated. The lazy
  draw-time lookup is the load-bearing detail that makes Mode 1 observable
  even for Plottables constructed before the reassignment.
- **D-07 mpl-widgets over ipywidgets** — `matplotlib.widgets` keep
  InteractivePanel controls inside the figure canvas with consistent
  styling. The pivot was driven by user verification at 09-04's checkpoint.
- **D-08 incremental notebook buildout** — cell 01 ships as a 1×3 mixed Figure
  for §2.4 Figure 1 only; subsequent cells (09-07+) layer in additional
  §2.4 figures as the user's prose authoring proceeds. Phase 9 ships when
  09-06 lands; the rest of the §2.4 figures are explicitly out-of-scope for
  Phase 9 (deferred to follow-up plans).

## Self-Check

Created/modified files verified on disk:

| File | Exists |
|---|---|
| `src/subshader/dsp/dsp.ipynb` | FOUND (cell 7ab635e8 populated) |
| `research/tests/dsplot/test_dsplot_example.py` | FOUND |
| `research/dsplot/figures/style_override_demo.py` | FOUND |
| `.planning/phases/09-dsplot-library-architecture/style-override-snippet.md` | FOUND |
| `research/dsp_figures.py` (63 lines, shim) | FOUND |
| `assets/images/dsp/style_override_demo/default_palette.png` | FOUND |
| `assets/images/dsp/style_override_demo/global_override.png` | FOUND |
| `assets/images/dsp/style_override_demo/local_override.png` | FOUND |
| `assets/images/diagnostics/09-06-checkpoint/cell01_full_render.png` | FOUND |

Commits verified in `git log`:

| Hash | Subject |
|---|---|
| `743fd22` | feat(09-06): populate dsp.ipynb cell 01 with §2.4 Figure 1 as 1x3 mixed Figure (D-08) |
| `5cd2c51` | test(09-06): add dsplot example integration test (teaching artifact) |
| `a0bbe61` | feat(09-06): style-override demo + docs snippet for BOTH D-05 modes |
| `6703556` | refactor(09-06): retire research/dsp_figures.py to a deprecation shim |

Verification commands all pass:

- `python -m pytest research/tests/dsplot/ -q` → **110 passed, 5 warnings**
- `grep -rIn "subshader" research/dsplot/ --exclude-dir=figures` → **zero matches** (D-01)
- `wc -l research/dsp_figures.py` → **63** (< 70 acceptance threshold)
- `PYTHONPATH=research python -W error::DeprecationWarning -c "from research.dsp_figures import A"` → **exits non-zero** (DeprecationWarning fires)
- `PYTHONPATH=research python -c "from research.dsp_figures import A, MOTIVATOR_VERSIONS, generate_all_dsp_figures; print(A, len(MOTIVATOR_VERSIONS))"` → **prints `(2.0, 3.0) 6`** (aliases work)
- `cd research && python -m dsplot.figures.style_override_demo` → produces three PNGs, restores global default
- `python -c "import nbformat; nb = nbformat.read('src/subshader/dsp/dsp.ipynb', as_version=4); cell = [c for c in nb.cells if c.get('id')=='7ab635e8'][0]; assert 'DynamicPanel' in cell.source and 'StaticPanel' in cell.source and 'A_PRIME_X_SWEEP' in cell.source"` → **passes**

## Self-Check: PASSED

## Awaiting

This plan ends in a `checkpoint:human-verify`. The full 6-target verification
sweep is documented in the plan's Task 4 `<how-to-verify>` block. The notebook
cell 01 animation behavior is the highest-signal item (Target #3) — the user
opens `src/subshader/dsp/dsp.ipynb` in Jupyter, runs cell 01 under
`%matplotlib widget`, and confirms the three-panel composition animates as
described. After all 6 targets verify, Phase 9 ships and the optional
`_09_05`-suffixed parity PNGs from 09-05 can be renamed to drop the suffix.
