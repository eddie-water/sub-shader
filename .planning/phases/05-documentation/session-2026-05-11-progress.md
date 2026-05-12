---
workstream: §2.4 Vector Projection prose + figure palette refactor
status: 2.4.1 prose reordered; 2D + 3D figures palette-unified to "Combo 5"; ready for prose authoring tomorrow
last_updated: 2026-05-11
session_type: local; pickup is remote tomorrow
---

# Session progress — 2026-05-11

## What landed today

### §2.4 Vector Projection — structural reorder
Goal was: nail down 2.4 prose, consolidate x/y projection with the symmetry beat, bridge to 3D + N dimensions, and set up §2.5 (sign accumulator).

- **§2.4.1 reflowed** so each figure leads the bullets that *read* it (instead of 5 abstract bullets sitting between section header and figure 1):
  - Pre-figure-1: 2 definitional bullets (vector = magnitude+direction; dot product = projection / shadow / component).
  - Figure 1 (`projection_reference_directions`).
  - Post-figure-1: a-onto-b bullet, the symmetry bullet (still `[]` placeholder — user authoring), the geometric-trust bullet + 3blue1brown video link.
  - Figure 2 (`dot_product_geometry`).
  - Post-figure-2: magnitude / 3-cases bullet.
  - WRITE markers stayed adjacent to the bullets they describe.
- **No prose was rewritten.** Only the order changed. The `[]` placeholder at line 127 (the symmetry sentence) is still the user's to author — three candidate one-liners workshopped during the session live in the chat history (Option A/B/C/D for symmetry bullet; Option 1/2/3 for magnitude+sign consolidation if user wants to expand the 3-cases bullet).

### Figure palette — unified Combo 5 across 2D + 3D
User locked the trio:
- **PRIMARY**   = `#e1641a` — saturated warm orange
- **SECONDARY** = `#7b6fe1` — medium violet
- **TERTIARY**  = `#ffd27d` — soft warm gold

User's pedagogical contract for color:
- Role colors paint **meaning** — vector identity in 2D, dimension identity in 3D.
- Spines, droplines, axis crosshairs, arrow labels — all **neutral**.
- Projection results (components in P1, shadows in P2/P3) are **shadows, shadows are neutral**.
- TERTIARY gold is **reserved for the third dimension** — surfaces only when 3D enters.

Mapping that fell out:
| Element | 2D figure | 3D figure |
|---|---|---|
| Vector `a`            | PRIMARY orange   | neutral off-white (no conflict with x-component) |
| Vector `b`            | SECONDARY purple | n/a |
| 1st-dim component (x) | neutral (shadow) | PRIMARY orange |
| 2nd-dim component (y) | neutral (shadow) | SECONDARY purple |
| 3rd-dim component (z) | n/a              | TERTIARY gold |
| Spines / droplines / labels | neutral grey | neutral grey |

### Architecture change — palette hoisted to `style.py`
- Three new constants in `research/utilities/style.py`:
  ```python
  PALETTE_PRIMARY   = "#e1641a"
  PALETTE_SECONDARY = "#7b6fe1"
  PALETTE_TERTIARY  = "#ffd27d"
  ```
- `VECTOR_A_COLOR` / `VECTOR_B_COLOR` aliased to `PALETTE_PRIMARY` / `PALETTE_SECONDARY`.
- `VECTOR_NEUTRAL_COLOR` updated to `#EEEEEE` (off-white).
- The local `_VECTOR_DIM_*` block in `dsp_figures.py` was deleted; replaced with thin aliases (`_DIM_X_COLOR`, `_DIM_Y_COLOR`, `_DIM_Z_COLOR`, `_DIM_NEUTRAL`, `_DIM_SPINE`) that pull from `style.PALETTE_*`. Single source of truth — the 2D and 3D figures now share the same palette source.
- **Bug found and fixed in passing**: 3D figure was referencing `_VECTOR_3D_*` constants that no longer existed (rename leftover from prior session). Would have crashed on next invocation. Now wired through the new aliases.

### Math annotations moved out of figure → markdown LaTeX
The 2D figure used to carry monospace math text under each panel (`a · b = (1.0)(0.6) + (0.5)(0.8) = 1.00`). Replaced with:
- Figure becomes visual-only (panel titles + arrow labels only).
- Math now lives as `$$...$$` LaTeX blocks under the figure in DSP.md, matching §2.3's `\vec{a}` style.
- Three-line block: vector definitions, `a · b` expansion, `b · a` expansion. Symbolic form (`a_x b_x + a_y b_y`) → plug in numbers → arithmetic → result.

### Docstring elaboration
Both figure functions now carry full docstrings explaining the pedagogical intent, panel-by-panel breakdown, color palette mapping, and why each color choice is what it is:
- `_plot_projection_reference_directions` (2D, §2.4.1)
- `_plot_vector_projection_3d` (3D, §2.4.2)

## Files touched

| File | Change |
|---|---|
| `src/subshader/dsp/DSP.md` | §2.4.1 bullets reordered (figure-leads-bullets); LaTeX math block added under projection figure; new figure filenames |
| `research/dsp_figures.py` | `_plot_projection_reference_directions` refactored to combo-5 palette; `_plot_vector_projection_3d` constants fixed; both functions' docstrings expanded |
| `research/utilities/style.py` | Added `PALETTE_PRIMARY/SECONDARY/TERTIARY`; aliased `VECTOR_A_COLOR`/`VECTOR_B_COLOR`; updated `VECTOR_NEUTRAL_COLOR` |
| `research/utilities/plotting.py` | (from earlier in session) `setup_vector_axes` accepts per-axis `x_color`/`y_color` overrides |

## New PNGs generated (committed)

Per the never-overwrite-PNGs policy, each iteration got a new descriptive filename:
- `assets/images/dsp/projection_reference_directions_v6_combo5_palette.png` ← **current canonical**
- `assets/images/dsp/vector_projection_3d_v2_combo5_palette.png` ← **current canonical**

Iteration history (kept for reference):
- `projection_reference_directions.png` — original (pre-recolor)
- `projection_reference_directions_v2_components_consistent.png` — vectors unified across panels
- `projection_reference_directions_v3_ascii_subscript.png` — fixed missing-glyph issue
- `projection_reference_directions_v4_math_in_md.png` — math stripped from figure
- `projection_reference_directions_v5_default_height.png` — figure shrunk back to default height
- `projection_reference_directions_v6_combo5_palette.png` — combo-5 palette applied

## Open / pending for tomorrow

### §2.4.1 prose authoring
Three placeholders still need user authoring (the structural skeleton is locked):

1. **Line 127** — symmetry bullet: `Notice how when we project **a** onto **b** or **b** onto **a**, the resulting __ [...]`
   Workshopped candidates from chat (pick + adapt):
   - **A** (terse): `Notice how projecting a onto b lands at the same dot product as projecting b onto a — the reference direction is a choice, not a constraint.`
   - **C** (introduces "symmetric", sets up §2.5): `Notice how projecting a onto b and projecting b onto a collapse to the same scalar. The dot product is symmetric — neither vector is privileged as the reference.`

2. **Line 142-145** — magnitude / 3-cases bullet currently lists 3 cases; WRITE marker for beat 5 wants 4 cases (adds "oblique"). Workshopped two-bullet consolidations (pick + adapt):
   - **Option 1** (magnitude + sign organizing axes — sets up §2.5 directly): two bullets, one for magnitude, one for sign.
   - **Option 2** (extremes + in-between): closer to how the 4-panel figure reads.

3. **Line 138** — geometric-trust bullet: still in `[]` form. Reads `[This actually works because of the symmetry found the geometry of the triangle these two vectors make ...]` → user prose.

### §2.4.2 prose authoring (next, not yet started)
Two WRITE markers in DSP.md beneath the 3D figure:
- Beat 1 — 3D extension prose (figure already explains visually; need short text).
- Beat 2 — N-dimensional reframe with the **locked bridge sentence**:
  > "Notice how the pattern in two dimensions applied to three dimensions, and the pattern expands to any number of n — but we can't really visualize n dimensions, so we'll drop that terminology and think of it in terms of N pair-wise multiplications whose sign agreements accumulate into one running total."

### §2.5 (Sign Accumulation) — not started
The locked bridge sentence in §2.4.2 explicitly hands off to "N pair-wise multiplications whose sign agreements accumulate into one running total" — that's §2.5's thesis statement. Need plan + figures.

### Out-of-scope cleanups noticed this session
- Old palette-design-progress.md `style.py` migration plan is now partially complete — three palette constants are added, but `VECTOR_A_COLOR` / `VECTOR_B_COLOR` were aliased rather than fully migrated. Other VECTOR_* constants in style.py still hold legacy hex (`VECTOR_PROJ_COLOR = "#ff8c66"` is now legacy; new figures render projections as neutral). Either deprecate explicitly or leave as legacy with a comment — user's call.

## How to resume in the remote session tomorrow

1. Pull the branch: `git checkout gsd/phase-08-codebase-refactoring-and-module-cleanup && git pull`
2. Open `src/subshader/dsp/DSP.md` — start at line 127 (the §2.4.1 symmetry bullet placeholder).
3. Inspect the canonical figures:
   - `assets/images/dsp/projection_reference_directions_v6_combo5_palette.png`
   - `assets/images/dsp/vector_projection_3d_v2_combo5_palette.png`
4. Per `feedback_remote_session_push_policy.md`: free to push markdown + PNG iterations without confirmation; code changes still need explicit confirmation.
5. Per `feedback_png_naming.md`: every figure regen gets a new descriptive filename — never overwrite (mobile caches by URL).
