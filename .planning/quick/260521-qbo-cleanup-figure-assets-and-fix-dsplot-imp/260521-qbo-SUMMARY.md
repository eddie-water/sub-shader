---
phase: 260521-qbo
plan: 01
type: quick
date: 2026-05-21
files_changed: 19  # 4 source modules + DSP.md + dsp.ipynb + 13 other rewrites; 175 PNGs moved + 1 regenerated
pngs_moved: 175
pngs_deleted: 0
families_created: 21
committed: false  # user reviews and commits manually (no-auto-commits policy)
---

# Quick 260521-qbo — Cleanup Figure Assets and Fix dsplot Imports — SUMMARY

## Overview

Reorganized 175 loose PNGs from `assets/images/{generated,dsp}/` into 21 per-family
subdirectories under `assets/images/figures/<family>/<name>.png`, stripping legacy
prefixes (`figure_1_`, `dsp_motivator_`, `lego_demo_`, etc.). Rewrote 14 broken
`from dsplot import …` lines across `research/dsplot/figures/*.py` to use the
relative form `from .. import …`, so every figure module now imports cleanly from
the repo root with no PYTHONPATH hack. Updated all render-function `output_dir` /
`output_filename` default kwargs to point at the new tree, updated 5 DSP.md image
links + 4 dsp.ipynb image cell refs, and re-rendered figure_1 + components_recombine
to verify the new pipeline writes only to the new tree.

## Decisions made

- **`style_override_demo.py`: converted to relative imports + dropped sys.path bootstrap.**
  The file is invoked as `python -m research.dsplot.figures.style_override_demo`, so
  the package context already makes relative imports resolve — the legacy
  `sys.path.insert(0, _RESEARCH_DIR)` block was redundant. Replaced
  `import dsplot` + `from dsplot import …` with `from .. import Figure, StaticPanel,
  Vector, style`, and rewrote every `dsplot.style.PRIMARY_COLOR` reference to
  `style.PRIMARY_COLOR` (mutating the shared `style` module object is what Mode 1
  global-override demonstrates — semantics unchanged).

- **`utilities` imports also rewritten to `research.utilities` (Rule 3 auto-fix).**
  Not in the plan, but module-load smoke-test failed without it (pre-existing
  parallel issue to the dsplot one). 5 occurrences across `figure_1.py`,
  `motivator.py`, `alignment_diagnostic.py` rewritten from `from utilities import`
  to `from research.utilities import`. Same pattern as the dsplot fix — package
  is a namespace package, absolute import works from repo root, no PYTHONPATH
  needed.

- **`baseline.png` for zero-suffix collisions.** Files whose stripped name would
  be empty (e.g. `style_skeleton.png` → `style_skeleton/.png`) land as
  `<family>/baseline.png`. Affects: `alignment_diagnostic/baseline.png`,
  `comparison_grid/baseline.png`, `lego/baseline.png`,
  `projection_reference_directions/baseline.png`, `style_skeleton/baseline.png`,
  `vector_basics/baseline.png`, `vector_projection/baseline.png`,
  `vector_projection_3d/baseline.png`, `vector_similarity/baseline.png`,
  `vector_symmetric_projection/baseline.png`, `vector_xy_projection/baseline.png`,
  `vector_xy_reconstruction/baseline.png`.

- **lego_demo source dir = `assets/images/dsp/`** (filesystem-confirmed; the
  plan's `<inventory>` block's "lego in generated/" note was outdated). All 47
  lego PNGs moved from `dsp/` to `figures/lego/`.

- **`misc/` family for ungrouped one-offs.** `20-20k is bent.png`,
  `beltran_sc_rip_16_bar.png`, `musical_signal_comparison.png`,
  `polyphonic_signal_comparison.png`, `timing_bar_chart.png` — these don't fit
  any other family. Kept their original stems unchanged.

- **No `git commit` executed.** Per user policy [[feedback-no-auto-commits]],
  all 226 changes left staged in the working tree for user review.

## Files modified

### Source modules (Tasks 2 + 3)

| File | What changed |
|---|---|
| `research/dsplot/figures/figure_1.py` | `from dsplot import` → `from ..`; `from utilities import` → `from research.utilities import` (2 places); `render()` / `render_hero()` / `render_antihero()` defaults → `assets/images/figures/figure_1/` + stripped filenames |
| `research/dsplot/figures/components_recombine.py` | `from dsplot import` → `from ..`; `render()` default filename → `either_order_v19.png` (prefix stripped) |
| `research/dsplot/figures/lego_demo.py` | `from dsplot import` → `from ..` |
| `research/dsplot/figures/style_skeleton.py` | `from dsplot import` → `from ..` |
| `research/dsplot/figures/motivator.py` | `from dsplot import` → `from ..`; `from utilities import` → `from research.utilities import` (2 places); 7 × `MotivatorConfig.output_filename` literals stripped of `dsp_motivator_` prefix |
| `research/dsplot/figures/vector_basics.py` | `from dsplot import` → `from ..` |
| `research/dsplot/figures/vector_projection_3d.py` | `from dsplot import …` → `from ..`; `from dsplot.panels import` → `from ..panels import` |
| `research/dsplot/figures/dot_product_geometry.py` | `from dsplot import` → `from ..` |
| `research/dsplot/figures/projection_reconstruction.py` | `from dsplot import` → `from ..` |
| `research/dsplot/figures/alignment_diagnostic.py` | `from dsplot import` → `from ..`; `from utilities import` → `from research.utilities import` |
| `research/dsplot/figures/style_override_demo.py` | Dropped sys.path bootstrap + absolute `import dsplot`; `from dsplot import` → `from ..`; replaced `dsplot.style.X` → `style.X` throughout |
| `research/dsplot/figures/__main__.py` | `_OUT_DIR` → `assets/images/figures/`; motivator_out → `assets/images/figures/motivator/`; 7 per-call output_filename strings rewritten with family-dir prefix; docstring + `--out` help updated |
| `research/dsplot/figures/__init__.py` | Docstring path updated to `assets/images/figures/<family>/` |

### Docs (Task 4)

| File | What changed |
|---|---|
| `src/subshader/dsp/DSP.md` | 5 image links rewritten from `../../../assets/images/dsp/<flat>.png` to `../../../assets/images/figures/<family>/<name>.png` (lines 130, 140, 169, 199, 245) |
| `src/subshader/dsp/dsp.ipynb` | 4 image cell refs rewritten via JSON load/dump (preserves nbformat structure) — projection_reconstruction, dot_product, vector_projection_3d, vector_xy_reconstruction |

### Assets (Task 1)

- 51 tracked PNGs moved via `git mv` (rename history preserved)
- 124 untracked PNGs moved via `mv` + `git add`
- 21 new family subdirs created under `assets/images/figures/`
- 0 PNG deletions in staged changes
- Pre-existing organized subdirs left untouched: `assets/images/dsp/style_override_demo/`, `assets/images/generated/stubs/`, `assets/images/generated/dpi/`

## How to run each render

All from repo root, no PYTHONPATH needed:

```bash
# Whole pipeline (writes all DSP.md figures into per-family subdirs)
python -m research.dsplot.figures

# Whole pipeline with a coexistence suffix (--suffix gets inserted before .png)
python -m research.dsplot.figures --suffix _$(date +%m_%d)

# Figure 1 family individually (locked v8 chirp pipeline; do not edit logic)
python -c "from research.dsplot.figures.figure_1 import render; print(render())"
python -c "from research.dsplot.figures.figure_1 import render_hero; print(render_hero())"
python -c "from research.dsplot.figures.figure_1 import render_antihero; print(render_antihero())"

# Style override demo (Mode 1 + Mode 2 side-by-side)
python -m research.dsplot.figures.style_override_demo

# Motivator family (all VERSIONS configs)
python -c "
from research.dsplot.figures.motivator import render_all
for p in render_all('assets/images/figures/motivator'):
    print(p)
"
```

## What would be committed (preview)

The orchestrator will not commit per the user rule; here is a one-line summary
the user can use to decide commit scope:

```
qbo-260521 staged changes:
  175 PNGs relocated under assets/images/figures/<family>/<name>.png (51 R via git mv, 124 A via mv+add)
  1 PNG re-rendered (figures/figure_1/fourier_vs_wavelet.png — modified after move) + 1 re-render-verify artifact (figures/components_recombine/either_order_v19_reverify.png)
  13 .py edits across research/dsplot/figures/
  1 DSP.md (5 image-link rewrites)
  1 dsp.ipynb (4 image-ref rewrites)
```

## Verification log

All checks passed before the human-verify checkpoint:

1. `grep -rn "^from dsplot\|^import dsplot" research/dsplot/figures/` → 0 hits
2. `git status --short -- assets/images/dsp/ assets/images/generated/` → 51 D entries (= the source half of the 51 git mv renames; the destination half is 51 R entries in the full status). 0 unrelated deletions in assets.
3. `python -c "from research.dsplot.figures.figure_1 import render, render_hero, render_antihero"` → imports OK
4. `python -c "from research.dsplot.figures import {figure_1, components_recombine, lego_demo, style_skeleton, motivator, vector_basics, vector_projection_3d, dot_product_geometry, projection_reconstruction, alignment_diagnostic, style_override_demo}"` → all 11 modules import cleanly
5. `render(), render_hero(), render_antihero()` → printed paths all under `assets/images/figures/figure_1/` (NOT under `assets/images/generated/`); files exist on disk
6. `components_recombine.render(...)` → printed path under `assets/images/figures/components_recombine/`
7. DSP.md image links → all 5 resolve (file -f check)
8. dsp.ipynb image refs → all 4 resolve (4 unique refs; JSON re-parses successfully)
9. `git log --oneline -3` → no executor commit (top commit is `a4e5f65 updated style template to v52 - spacing and colors` from prior session)

## Self-Check: PASSED

- Plan file: `.planning/quick/260521-qbo-cleanup-figure-assets-and-fix-dsplot-imp/260521-qbo-PLAN.md` exists
- All 175 PNGs in mapping table accounted for under `assets/images/figures/`
- All 13 source-module edits + DSP.md + dsp.ipynb edits applied
- Verification checklist from CRITICAL_OVERRIDE: 4/4 passed (broken-imports grep empty; renames staged as R; renders write to new tree; no executor commits)
- No `git commit` was executed by this executor

## Old to New PNG Mapping (175 files)

Grouped by destination family. Tracked files were moved via `git mv` (history preserved); untracked were `mv`'d + `git add`'d.

### `alignment_diagnostic/` (4 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/dsp_alignment_diagnostic.png` | `figures/alignment_diagnostic/baseline.png` | no |
| `generated/dsp_alignment_diagnostic_09_05.png` | `figures/alignment_diagnostic/09_05.png` | yes |
| `generated/dsp_alignment_diagnostic_2_weird_chunk.png` | `figures/alignment_diagnostic/2_weird_chunk.png` | no |
| `generated/dsp_alignment_diagnostic_BEFORE_FIX.png` | `figures/alignment_diagnostic/BEFORE_FIX.png` | no |

### `chirp/` (3 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/chirp_random_walk.png` | `figures/chirp/random_walk.png` | yes |
| `generated/chirp_signal_comparison.png` | `figures/chirp/signal_comparison.png` | yes |
| `generated/chirp_sweep_200_20k.png` | `figures/chirp/sweep_200_20k.png` | yes |

### `comparison_grid/` (4 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/comparison_grid.png` | `figures/comparison_grid/baseline.png` | yes |
| `generated/comparison_grid_STUB.png` | `figures/comparison_grid/STUB.png` | yes |
| `generated/comparison_grid_STUB_PYWT.png` | `figures/comparison_grid/STUB_PYWT.png` | yes |
| `generated/comparison_grid_chunksize16k.png` | `figures/comparison_grid/chunksize16k.png` | yes |

### `components_recombine/` (20 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/components_recombine_either_order.png` | `figures/components_recombine/either_order.png` | no |
| `dsp/components_recombine_either_order_v10_clean_dashed_heads.png` | `figures/components_recombine/either_order_v10_clean_dashed_heads.png` | no |
| `dsp/components_recombine_either_order_v11_a_and_aprime.png` | `figures/components_recombine/either_order_v11_a_and_aprime.png` | no |
| `dsp/components_recombine_either_order_v12_style_standardized.png` | `figures/components_recombine/either_order_v12_style_standardized.png` | no |
| `dsp/components_recombine_either_order_v13_bigger_numbers.png` | `figures/components_recombine/either_order_v13_bigger_numbers.png` | no |
| `dsp/components_recombine_either_order_v14_spine_pm4.png` | `figures/components_recombine/either_order_v14_spine_pm4.png` | no |
| `dsp/components_recombine_either_order_v15_label_offset.png` | `figures/components_recombine/either_order_v15_label_offset.png` | no |
| `dsp/components_recombine_either_order_v16_a_4_3.png` | `figures/components_recombine/either_order_v16_a_4_3.png` | no |
| `dsp/components_recombine_either_order_v17_a_2_3.png` | `figures/components_recombine/either_order_v17_a_2_3.png` | no |
| `dsp/components_recombine_either_order_v18.png` | `figures/components_recombine/either_order_v18.png` | no |
| `dsp/components_recombine_either_order_v18_09_05.png` | `figures/components_recombine/either_order_v18_09_05.png` | yes |
| `dsp/components_recombine_either_order_v19.png` | `figures/components_recombine/either_order_v19.png` | no |
| `dsp/components_recombine_either_order_v2_canonical_numbers.png` | `figures/components_recombine/either_order_v2_canonical_numbers.png` | no |
| `dsp/components_recombine_either_order_v3_3panels.png` | `figures/components_recombine/either_order_v3_3panels.png` | no |
| `dsp/components_recombine_either_order_v4_3panels_dashed.png` | `figures/components_recombine/either_order_v4_3panels_dashed.png` | no |
| `dsp/components_recombine_either_order_v5_perpendicular_panel.png` | `figures/components_recombine/either_order_v5_perpendicular_panel.png` | no |
| `dsp/components_recombine_either_order_v6_panel2_full_labels.png` | `figures/components_recombine/either_order_v6_panel2_full_labels.png` | no |
| `dsp/components_recombine_either_order_v7_two_thirds_ratio.png` | `figures/components_recombine/either_order_v7_two_thirds_ratio.png` | no |
| `dsp/components_recombine_either_order_v8_sharp_arrows.png` | `figures/components_recombine/either_order_v8_sharp_arrows.png` | no |
| `dsp/components_recombine_either_order_v9_shorter_heads.png` | `figures/components_recombine/either_order_v9_shorter_heads.png` | no |

### `dot_product/` (5 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/dot_product_geometry.png` | `figures/dot_product/geometry.png` | yes |
| `dsp/dot_product_geometry_09_05.png` | `figures/dot_product/geometry_09_05.png` | yes |
| `dsp/dot_product_symmetry.png` | `figures/dot_product/symmetry.png` | no |
| `dsp/dot_product_twin_rectangles_v1.png` | `figures/dot_product/twin_rectangles_v1.png` | no |
| `dsp/dot_product_twin_rectangles_v2.png` | `figures/dot_product/twin_rectangles_v2.png` | no |

### `figure_1/` (17 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/figure_1_fourier_vs_wavelet.png` | `figures/figure_1/fourier_vs_wavelet.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v2_dsplot_v10_style.png` | `figures/figure_1/fourier_vs_wavelet_v2_dsplot_v10_style.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v3_inst_freq_overlay.png` | `figures/figure_1/fourier_vs_wavelet_v3_inst_freq_overlay.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v4_primary_inst_freq.png` | `figures/figure_1/fourier_vs_wavelet_v4_primary_inst_freq.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v5_thicker_inst_freq.png` | `figures/figure_1/fourier_vs_wavelet_v5_thicker_inst_freq.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v6_gray_inst_freq.png` | `figures/figure_1/fourier_vs_wavelet_v6_gray_inst_freq.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v7_orange_curve_gray_chirp.png` | `figures/figure_1/fourier_vs_wavelet_v7_orange_curve_gray_chirp.png` | no |
| `generated/figure_1_fourier_vs_wavelet_v8_inst_freq_on_all_rows.png` | `figures/figure_1/fourier_vs_wavelet_v8_inst_freq_on_all_rows.png` | no |
| `generated/figure_1_hero_candidate_2_v14_20ms_spacing.png` | `figures/figure_1/hero_candidate_2_v14_20ms_spacing.png` | no |
| `generated/figure_1_hero_candidate_2_v15_bigger_fonts.png` | `figures/figure_1/hero_candidate_2_v15_bigger_fonts.png` | no |
| `generated/figure_1_hero_candidate_2_v16_thick_inst_freq.png` | `figures/figure_1/hero_candidate_2_v16_thick_inst_freq.png` | no |
| `generated/figure_1_hero_candidate_2_v18_mobile.png` | `figures/figure_1/hero_candidate_2_v18_mobile.png` | no |
| `generated/figure_1_hero_candidate_2_v18_wider_labels_centered_sup.png` | `figures/figure_1/hero_candidate_2_v18_wider_labels_centered_sup.png` | no |
| `generated/figure_1_hero_v41_style_v1.png` | `figures/figure_1/hero_v41_style_v1.png` | no |
| `generated/figure_1_hero_v41_style_v2_cell_borders.png` | `figures/figure_1/hero_v41_style_v2_cell_borders.png` | no |
| `generated/figure_1_hero_v41_style_v3_twin_xtick_fix.png` | `figures/figure_1/hero_v41_style_v3_twin_xtick_fix.png` | no |
| `generated/figure_1_hero_v41_style_v4_inst_freq_halo.png` | `figures/figure_1/hero_v41_style_v4_inst_freq_halo.png` | no |

### `lego/` (47 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/lego_demo.png` | `figures/lego/baseline.png` | yes |
| `dsp/lego_demo_current_state.png` | `figures/lego/current_state.png` | no |
| `dsp/lego_demo_post_xtick_fix.png` | `figures/lego/post_xtick_fix.png` | no |
| `dsp/lego_demo_v10.png` | `figures/lego/v10.png` | yes |
| `dsp/lego_demo_v11.png` | `figures/lego/v11.png` | no |
| `dsp/lego_demo_v12.png` | `figures/lego/v12.png` | no |
| `dsp/lego_demo_v13.png` | `figures/lego/v13.png` | no |
| `dsp/lego_demo_v14.png` | `figures/lego/v14.png` | no |
| `dsp/lego_demo_v15.png` | `figures/lego/v15.png` | no |
| `dsp/lego_demo_v16.png` | `figures/lego/v16.png` | no |
| `dsp/lego_demo_v17.png` | `figures/lego/v17.png` | no |
| `dsp/lego_demo_v18.png` | `figures/lego/v18.png` | no |
| `dsp/lego_demo_v19.png` | `figures/lego/v19.png` | no |
| `dsp/lego_demo_v20.png` | `figures/lego/v20.png` | no |
| `dsp/lego_demo_v21.png` | `figures/lego/v21.png` | no |
| `dsp/lego_demo_v22.png` | `figures/lego/v22.png` | no |
| `dsp/lego_demo_v23.png` | `figures/lego/v23.png` | no |
| `dsp/lego_demo_v24.png` | `figures/lego/v24.png` | no |
| `dsp/lego_demo_v25.png` | `figures/lego/v25.png` | no |
| `dsp/lego_demo_v26.png` | `figures/lego/v26.png` | no |
| `dsp/lego_demo_v27.png` | `figures/lego/v27.png` | no |
| `dsp/lego_demo_v28.png` | `figures/lego/v28.png` | no |
| `dsp/lego_demo_v29.png` | `figures/lego/v29.png` | no |
| `dsp/lego_demo_v30.png` | `figures/lego/v30.png` | no |
| `dsp/lego_demo_v31.png` | `figures/lego/v31.png` | no |
| `dsp/lego_demo_v32.png` | `figures/lego/v32.png` | no |
| `dsp/lego_demo_v33.png` | `figures/lego/v33.png` | no |
| `dsp/lego_demo_v34.png` | `figures/lego/v34.png` | no |
| `dsp/lego_demo_v35.png` | `figures/lego/v35.png` | no |
| `dsp/lego_demo_v36.png` | `figures/lego/v36.png` | no |
| `dsp/lego_demo_v37.png` | `figures/lego/v37.png` | no |
| `dsp/lego_demo_v38.png` | `figures/lego/v38.png` | no |
| `dsp/lego_demo_v39.png` | `figures/lego/v39.png` | no |
| `dsp/lego_demo_v40.png` | `figures/lego/v40.png` | no |
| `dsp/lego_demo_v41.png` | `figures/lego/v41.png` | yes |
| `dsp/lego_demo_v42.png` | `figures/lego/v42.png` | no |
| `dsp/lego_demo_v43.png` | `figures/lego/v43.png` | no |
| `dsp/lego_demo_v44.png` | `figures/lego/v44.png` | no |
| `dsp/lego_demo_v45.png` | `figures/lego/v45.png` | no |
| `dsp/lego_demo_v46.png` | `figures/lego/v46.png` | no |
| `dsp/lego_demo_v47.png` | `figures/lego/v47.png` | no |
| `dsp/lego_demo_v48.png` | `figures/lego/v48.png` | no |
| `dsp/lego_demo_v49.png` | `figures/lego/v49.png` | no |
| `dsp/lego_demo_v50.png` | `figures/lego/v50.png` | no |
| `dsp/lego_demo_v51.png` | `figures/lego/v51.png` | no |
| `dsp/lego_demo_v52.png` | `figures/lego/v52.png` | yes |
| `dsp/lego_demo_v8.png` | `figures/lego/v8.png` | yes |

### `misc/` (5 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/20-20k is bent.png` | `figures/misc/20-20k is bent.png` | no |
| `generated/beltran_sc_rip_16_bar.png` | `figures/misc/beltran_sc_rip_16_bar.png` | yes |
| `generated/musical_signal_comparison.png` | `figures/misc/musical_signal_comparison.png` | yes |
| `generated/polyphonic_signal_comparison.png` | `figures/misc/polyphonic_signal_comparison.png` | yes |
| `generated/timing_bar_chart.png` | `figures/misc/timing_bar_chart.png` | yes |

### `motivator/` (25 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/dsp_motivator_100-800hz_overlaid.png` | `figures/motivator/100-800hz_overlaid.png` | no |
| `generated/dsp_motivator_100-800hz_overlaid_red_boxes.png` | `figures/motivator/100-800hz_overlaid_red_boxes.png` | no |
| `generated/dsp_motivator_section1_20-20000hz_2.0s.png` | `figures/motivator/section1_20-20000hz_2.0s.png` | no |
| `generated/dsp_motivator_section1_200-20000hz_1.0s.png` | `figures/motivator/section1_200-20000hz_1.0s.png` | no |
| `generated/dsp_motivator_section1_200-20000hz_1.0s_waypoint.png` | `figures/motivator/section1_200-20000hz_1.0s_waypoint.png` | no |
| `generated/dsp_motivator_section1_200-20000hz_2.0s.png` | `figures/motivator/section1_200-20000hz_2.0s.png` | no |
| `generated/dsp_motivator_v1_50-250hz_0.2s.png` | `figures/motivator/v1_50-250hz_0.2s.png` | no |
| `generated/dsp_motivator_v2_50-500hz_0.3s.png` | `figures/motivator/v2_50-500hz_0.3s.png` | no |
| `generated/dsp_motivator_v3_100-800hz_0.4s.png` | `figures/motivator/v3_100-800hz_0.4s.png` | no |
| `generated/dsp_motivator_v4_100-2000hz_0.5s.png` | `figures/motivator/v4_100-2000hz_0.5s.png` | no |
| `generated/dsp_motivator_v4_100-2000hz_0.5s_09_05.png` | `figures/motivator/v4_100-2000hz_0.5s_09_05.png` | yes |
| `generated/dsp_motivator_v5_50-5000hz_1.0s.png` | `figures/motivator/v5_50-5000hz_1.0s.png` | no |
| `generated/dsp_motivator_v5_50-5000hz_1.0s_09_05.png` | `figures/motivator/v5_50-5000hz_1.0s_09_05.png` | yes |
| `generated/dsp_motivator_v6_20-20000hz_2.0s.png` | `figures/motivator/v6_20-20000hz_2.0s.png` | no |
| `generated/dsp_motivator_vw1_gentle_100-2000hz_0.5s.png` | `figures/motivator/vw1_gentle_100-2000hz_0.5s.png` | no |
| `generated/dsp_motivator_vw1_gentle_100-2000hz_0.5s_09_05.png` | `figures/motivator/vw1_gentle_100-2000hz_0.5s_09_05.png` | yes |
| `generated/dsp_motivator_vw2_moderate_100-2000hz_0.5s.png` | `figures/motivator/vw2_moderate_100-2000hz_0.5s.png` | no |
| `generated/dsp_motivator_vw2_moderate_100-2000hz_0.5s_09_05.png` | `figures/motivator/vw2_moderate_100-2000hz_0.5s_09_05.png` | yes |
| `generated/dsp_motivator_vw3_aggressive_100-2000hz_0.5s.png` | `figures/motivator/vw3_aggressive_100-2000hz_0.5s.png` | no |
| `generated/dsp_motivator_vw3_aggressive_100-2000hz_0.5s_09_05.png` | `figures/motivator/vw3_aggressive_100-2000hz_0.5s_09_05.png` | yes |
| `generated/dsp_motivator_vw4_aggressive_20-12000hz_1.4s.png` | `figures/motivator/vw4_aggressive_20-12000hz_1.4s.png` | no |
| `generated/dsp_motivator_vw4_aggressive_20-19000hz_1.4s.png` | `figures/motivator/vw4_aggressive_20-19000hz_1.4s.png` | no |
| `generated/dsp_motivator_vw4_aggressive_20-20000hz_1.0s.png` | `figures/motivator/vw4_aggressive_20-20000hz_1.0s.png` | no |
| `generated/dsp_motivator_vw4_aggressive_20-20000hz_2.0s.png` | `figures/motivator/vw4_aggressive_20-20000hz_2.0s.png` | yes |
| `generated/dsp_motivator_vw4_aggressive_20-20000hz_2.0s_09_05.png` | `figures/motivator/vw4_aggressive_20-20000hz_2.0s_09_05.png` | yes |

### `palette/` (8 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/palette_comparison_v1.png` | `figures/palette/comparison_v1.png` | no |
| `dsp/palette_lego_baseline_combo5.png` | `figures/palette/lego_baseline_combo5.png` | no |
| `dsp/palette_lego_hot_tangerine.png` | `figures/palette/lego_hot_tangerine.png` | no |
| `dsp/palette_lego_red_orange.png` | `figures/palette/lego_red_orange.png` | no |
| `dsp/palette_lego_saturated_mid.png` | `figures/palette/lego_saturated_mid.png` | no |
| `dsp/palette_lego_vivid_pure.png` | `figures/palette/lego_vivid_pure.png` | no |
| `dsp/palette_orange_comparison_10_v1.png` | `figures/palette/orange_comparison_10_v1.png` | yes |
| `dsp/palette_orange_comparison_v1.png` | `figures/palette/orange_comparison_v1.png` | no |

### `projection_reconstruction/` (10 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/projection_reconstruction_either_order.png` | `figures/projection_reconstruction/either_order.png` | no |
| `dsp/projection_reconstruction_either_order_v2_sharp_arrows.png` | `figures/projection_reconstruction/either_order_v2_sharp_arrows.png` | no |
| `dsp/projection_reconstruction_either_order_v3_styled_to_match_fig1.png` | `figures/projection_reconstruction/either_order_v3_styled_to_match_fig1.png` | no |
| `dsp/projection_reconstruction_either_order_v4_arrow_axes.png` | `figures/projection_reconstruction/either_order_v4_arrow_axes.png` | no |
| `dsp/projection_reconstruction_either_order_v5_b_1_4.png` | `figures/projection_reconstruction/either_order_v5_b_1_4.png` | no |
| `dsp/projection_reconstruction_either_order_v6_b_3_1.png` | `figures/projection_reconstruction/either_order_v6_b_3_1.png` | no |
| `dsp/projection_reconstruction_either_order_v7_a_4_3.png` | `figures/projection_reconstruction/either_order_v7_a_4_3.png` | no |
| `dsp/projection_reconstruction_either_order_v8_a_2_3.png` | `figures/projection_reconstruction/either_order_v8_a_2_3.png` | no |
| `dsp/projection_reconstruction_either_order_v9.png` | `figures/projection_reconstruction/either_order_v9.png` | no |
| `dsp/projection_reconstruction_either_order_v9_09_05.png` | `figures/projection_reconstruction/either_order_v9_09_05.png` | yes |

### `projection_reference_directions/` (6 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/projection_reference_directions.png` | `figures/projection_reference_directions/baseline.png` | yes |
| `dsp/projection_reference_directions_v2_components_consistent.png` | `figures/projection_reference_directions/v2_components_consistent.png` | yes |
| `dsp/projection_reference_directions_v3_ascii_subscript.png` | `figures/projection_reference_directions/v3_ascii_subscript.png` | yes |
| `dsp/projection_reference_directions_v4_math_in_md.png` | `figures/projection_reference_directions/v4_math_in_md.png` | yes |
| `dsp/projection_reference_directions_v5_default_height.png` | `figures/projection_reference_directions/v5_default_height.png` | yes |
| `dsp/projection_reference_directions_v6_combo5_palette.png` | `figures/projection_reference_directions/v6_combo5_palette.png` | yes |

### `sample_template/` (1 files)

| Old path | New path | Tracked? |
|---|---|---|
| `generated/sample_template_v1.png` | `figures/sample_template/v1.png` | no |

### `style_skeleton/` (9 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/style_skeleton.png` | `figures/style_skeleton/baseline.png` | yes |
| `dsp/style_skeleton_v12_clean.png` | `figures/style_skeleton/v12_clean.png` | no |
| `dsp/style_skeleton_v12_guides.png` | `figures/style_skeleton/v12_guides.png` | no |
| `dsp/style_skeleton_v13_clean.png` | `figures/style_skeleton/v13_clean.png` | no |
| `dsp/style_skeleton_v13_guides.png` | `figures/style_skeleton/v13_guides.png` | no |
| `dsp/style_skeleton_v3.png` | `figures/style_skeleton/v3.png` | yes |
| `dsp/style_skeleton_v8_clean.png` | `figures/style_skeleton/v8_clean.png` | yes |
| `dsp/style_skeleton_v8_guides.png` | `figures/style_skeleton/v8_guides.png` | yes |
| `dsp/style_skeleton_v9_guides.png` | `figures/style_skeleton/v9_guides.png` | yes |

### `vector_basics/` (2 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_basics.png` | `figures/vector_basics/baseline.png` | yes |
| `dsp/vector_basics_09_05.png` | `figures/vector_basics/09_05.png` | yes |

### `vector_projection/` (1 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_projection.png` | `figures/vector_projection/baseline.png` | yes |

### `vector_projection_3d/` (3 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_projection_3d.png` | `figures/vector_projection_3d/baseline.png` | yes |
| `dsp/vector_projection_3d_v2_combo5_palette.png` | `figures/vector_projection_3d/v2_combo5_palette.png` | yes |
| `dsp/vector_projection_3d_v2_combo5_palette_09_05.png` | `figures/vector_projection_3d/v2_combo5_palette_09_05.png` | yes |

### `vector_similarity/` (1 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_similarity.png` | `figures/vector_similarity/baseline.png` | yes |

### `vector_symmetric_projection/` (1 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_symmetric_projection.png` | `figures/vector_symmetric_projection/baseline.png` | yes |

### `vector_xy_projection/` (1 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_xy_projection.png` | `figures/vector_xy_projection/baseline.png` | yes |

### `vector_xy_reconstruction/` (2 files)

| Old path | New path | Tracked? |
|---|---|---|
| `dsp/vector_xy_reconstruction.png` | `figures/vector_xy_reconstruction/baseline.png` | yes |
| `dsp/vector_xy_reconstruction_09_05.png` | `figures/vector_xy_reconstruction/09_05.png` | yes |

## Open questions / follow-ups for next quick

- Should `dsplot` and `utilities` be promoted to top-level packages so the
  absolute imports (`from research.utilities import …`) become shorter
  (`from utilities import …` working everywhere)? Deferred — the plan
  explicitly forbade promotion, and the namespace-package form works fine.
- Should `assets/images/dsp/` and `assets/images/generated/` top-level dirs
  be removed (now empty at top level — only the protected subdirs remain)?
  Out of scope for qbo; would belong in a follow-up "directory tidy" quick.
- The 47 `lego/v*.png` files include many iterations from the v8→v52 style
  refinement series. A future quick could prune to canonical snapshots
  (v8, v41, v52) to reduce git history weight. Deferred.
- `assets/images/figures/` already contained 4 unrelated `*_edison.png` and
  `polyphonic-signal-example-midi-notes.png` files at the top level before
  this task; they were left untouched. Could be relocated into their own
  family in a follow-up.
