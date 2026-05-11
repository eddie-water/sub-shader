---
workstream: plot-lib palette design (sub-workstream of phase 5)
status: tier-system + nomenclature locked; accent family open; not yet applied to style.py
last_updated: 2026-05-11
scope: design exploration — no code changes to src/ or research/utilities/style.py yet
---

# Plot lib palette — design progress

## Why this exists

The custom plot lib for SubShader's pedagogical figures (`research/utilities/style.py` and friends) had organically accumulated color choices: yellow grid lines, cool-blue vector contrasts, peach projections, plus various `tab:orange`/`tab:blue` one-offs. Goal: standardize a small, intentional palette tied to the live CWT colormap's aesthetic, with a tier system that supports pedagogical "spotlight" patterns (the same figure rendered multiple times with different elements highlighted).

This work is **design exploration only** — no `style.py` edits made. Nothing in `src/` or `research/utilities/` has changed. Once the accent family is locked, the next step is to translate the decisions below into `style.py` constants.

## Locked decisions

### Nomenclature

**Tiers** (saturation level, tied to emphasis):

| Tier | Use |
|---|---|
| `SPOTLIGHT` | Saturated stop. The focused element of the panel. |
| `BASE` | Considered stop. Default rendering when nothing is singled out. |
| `MUTED` | Dusty stop. Present but de-emphasized; differentiated but quiet. |

**Roles** (which color family):

| Role | Family |
|---|---|
| `PRIMARY` | warm orange |
| `CONTRAST` | medium slate blue |
| `ACCENT` | tertiary — **not yet locked** (sage / rose / mustard / teal candidates) |

Constant naming: `<ROLE>_<TIER>` — e.g. `PRIMARY_SPOTLIGHT`, `CONTRAST_BASE`, `ACCENT_MUTED`. Nine constants once accent is picked.

### Hex values (locked roles)

```python
PRIMARY_SPOTLIGHT  = "#F5A83D"   # saturated warm orange
PRIMARY_BASE       = "#E79E55"   # considered
PRIMARY_MUTED      = "#E0A26C"   # dusty

CONTRAST_SPOTLIGHT = "#8878DD"   # saturated medium slate blue
CONTRAST_BASE      = "#9189D2"   # considered
CONTRAST_MUTED     = "#9C98CD"   # dusty
```

### Softening principle (for any future palette additions)

Three levers to soften a pure hue for dark-bg display:
1. **Drop saturation** — biggest lever. Pure 100% → ~50–65% S keeps the hue's character but removes the "alarm signal" feel.
2. **Avoid lightness extremes** — pure `#000`/`#FFF` are maximally contrasting against everything. Same principle that gives us `BG_COLOR = "#1A1A1A"` instead of pure black.
3. **Harmonize with bg temperature** — `#1A1A1A` is slightly warm. Foreground colors that "share air" with it (small hue shift toward warmer for cool colors, toward cooler for warm) feel cohesive instead of clashing.

Mental shortcut: **muted color = saturated hue mixed with a gray of similar lightness.**

### Inferno-fidelity tradeoff (decided)

The original goal was to source the highlight palette literally from the inferno colormap (the live CWT visualizer cmap). This works for the **warm half** (orange, gold, pink anchors all read well on dark bg) but fails for the **cool half** — inferno's deep purples `#420A68`/`#6A176E` are too dark to function as foreground highlights on `#1A1A1A`. The decision: warm side stays inferno-anchored; cool side steps outside the cmap to lilac/slate-blue territory that's bright enough for foreground use. The two halves are tonally compatible without being literally drawn from the same cmap.

## Open decisions

### 1. Accent family — pick ONE

Four candidates, all with three tiers computed:

| Family | SPOTLIGHT | BASE | MUTED | Color-theory role |
|---|---|---|---|---|
| Sage | `#52E081` | `#79D297` | `#9CCBB0` | True triadic with orange (30° + 150°) |
| Rose | `#F075A8` | `#DF90B1` | `#D4ABBC` | Split-complement; warm but distinct |
| Mustard | `#F2CC5A` | `#DBBD70` | `#CDBE98` | Analogous to orange — risk: reads as one fuzzy warm |
| Teal | `#47D1CD` | `#70C2C2` | `#96C0C0` | Analogous to slate blue — risk: competes with contrast |

User's prior read: **sage is the safest "third anchor"** (most distinct from primary + contrast). Rose is the prettiest but warm-leaning. Mustard probably weakest. Teal has a real risk of fighting the slate blue.

### 2. Tier-name confirmation

Current proposal: `SPOTLIGHT / BASE / MUTED`. Alternatives considered: `EMPHASIS / BASE / MUTED`, `LOUD / STANDARD / QUIET`, theatrical pairings like `SPOTLIGHT / WASH / SHADOW`. User adopted "spotlight" enthusiastically; `BASE / MUTED` are the working names for the other two tiers.

### 3. Apply to `style.py` — not yet done

`research/utilities/style.py` still has the old constants:
- `VECTOR_A_COLOR = "#fcfeaa"` (yellow)
- `VECTOR_B_COLOR = "#7ec8ff"` (cool blue)
- `VECTOR_PROJ_COLOR = "#ff8c66"` (peach)
- `GRID_WAVEFORM_COLOR = "#fcfeaa"` (yellow)
- `VECTOR_ORANGE = "tab:orange"` / `VECTOR_BLUE = "tab:blue"`

Once accent is locked, the work is:
1. Add the 9 `<ROLE>_<TIER>` constants to `style.py`.
2. Update existing semantic constants to map to the new tiers (e.g., `VECTOR_A_COLOR = PRIMARY_BASE`).
3. Decide whether to keep the existing semantic names as aliases or migrate call sites.

## Artifacts

### Rendering script (committed to repo)

- `.planning/phases/05-documentation/palette_schema_render.py` — single-file matplotlib script that produces the comprehensive schema PNG. Edit the `OUT` path and re-run to iterate. Run from repo root with `venv/bin/python .planning/phases/05-documentation/palette_schema_render.py`.

### Iteration PNGs (committed to `assets/images/claude/`)

Chronological order — keep all, the next session may want to compare:

| File | What it shows |
|---|---|
| `palette_preview.png` | Initial inferno inventory + N=1..8 scenarios |
| `palette_candidates.png` | First narrow 4-variant comparison (Variant A/B/C) |
| `palette_expanded.png` | Wider exploration; lilac/periwinkle/teal candidates introduced |
| `palette_softening.png` | Softening principles applied to slate-blue + warm-orange |
| `palette_tertiary.png` | Sage/rose/mustard/teal explored as tertiary |
| `palette_tiers.png` | First tier-system PNG (EMPHASIS/BASE/MUTED introduced) |
| `palette_schema.png` | First comprehensive schema (overwritten — has v2 content) |
| `palette_schema_v2_visible_projection.png` | Vector geometry: a (mag 1, 45°), b (mag 1.5, 0°), proj of b onto a |
| `palette_schema_v3_trios_and_a_on_b.png` | Adds trio-juxtaposition section; flips projection to a-onto-b |
| `palette_schema_v4_no_proj_label.png` | **LATEST.** Drops the "proj" text label from the projection vector |

### Vector example geometry (used in all schema PNGs)

```
a = mag 1, angle 45°   →  (0.707, 0.707)        # the smaller vector
b = mag 1.5, angle 0°  →  (1.5,   0)            # the longer vector
proj of a onto b       →  ~(0.707, 0)           # a's shadow on b
                                                  along b's direction
```

Dropline drops vertically from a's tip `(0.707, 0.707)` to the projection point `(0.707, 0)` — a clean perpendicular.

## How to resume

1. Open the latest PNG: `assets/images/claude/palette_schema_v4_no_proj_label.png`. Decide the accent family from the trio-juxtaposition rows.
2. If iterating visually further: edit `.planning/phases/05-documentation/palette_schema_render.py`, change `OUT` to a new descriptive filename (e.g. `palette_schema_v5_<change>.png`), run with `venv/bin/python`, copy to `assets/images/claude/`, commit, push.
3. When ready to apply to code: edit `research/utilities/style.py` per the migration plan in "Open decisions §3".

## Session conduct notes (saved to memory)

- `feedback_remote_session_push_policy.md` — In remote sessions: free to push md/png artifacts; code changes still require explicit confirmation.
- `feedback_png_naming.md` — Never overwrite existing PNGs (mobile caches by URL); each iteration gets a new descriptive filename.

Both apply to the next session.
