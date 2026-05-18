# dsplot Style Override — Two Modes (D-05)

Every visual default in dsplot lives on `dsplot.style` as an inheritable template
(colors, typography, panel sizing, layout spacing, arrowhead dimensions, …). Override
either GLOBALLY (affects every figure rendered after the reassignment) or LOCALLY
(single figure module only, global template untouched). Pick whichever matches
your intent.

## Mode 1: Global reassignment

Reassign the dsplot.style constants in place. Every figure rendered after the
reassignment picks up the new values — this is the "affects all figures" mode.

```python
import dsplot
from dsplot import Figure, StaticPanel, Vector

# Before: PRIMARY_COLOR is dsplot's default orange (#e1641a).
dsplot.style.PRIMARY_COLOR = "#22c55e"   # green

# Every figure rendered after this line uses #22c55e wherever a Plottable
# resolves None against style.PRIMARY_COLOR at draw() time.
fig = Figure(n_rows=1, n_cols=1)
fig.add_panel(StaticPanel(lim=2).add(Vector((1, 1), label="a")), row=0, col=0)
fig.render()
fig.savefig("green_vector.png")
fig.close()
```

**Use this when:** you want a project-wide theme change — light-mode swap,
brand palette substitution, dark-mode adjustment for a new audience, etc.

**Don't forget to restore the original if you do not want the change to persist
beyond your scope.** A `try/finally` around the reassignment keeps later code
unaffected.

## Mode 2: Local figure override

The figure module defines its OWN module-level constant derived from a default.
Other figures still inherit `dsplot.style.*` untouched.

```python
# in research/dsplot/figures/my_figure.py
from dsplot import style

# Local override: this figure wants a chunkier label gutter than the template.
# The derivation makes the override INTENT clear at the call site.
LABEL_RATIO = style.DEFAULT_LABEL_RATIO * 1.5

# Other figures still get style.DEFAULT_LABEL_RATIO (unchanged).
```

The production reference for this pattern is `research/dsplot/figures/motivator.py`,
where six layout constants (`LAYOUT_HSPACE`, `LAYOUT_LABEL_RATIO`,
`LAYOUT_MARGIN`, `ROW_LABEL_FONT_SIZE`, `AXIS_LABEL_FONT_SIZE`,
`TICK_LABEL_FONT_SIZE`) are all derived from `style.DEFAULT_*` so motivator can
diverge from the global template without rippling its change to other figures.

**Use this when:** ONE figure needs to differ from the template (e.g. a chunkier
title font for the README hero figure) without forcing every other figure to
change too. This is the most common override pattern in production.

## Why this works

Plottables resolve None-valued style knobs against `dsplot.style.*` at `draw()`
time (lazy lookup), not at construction time. That means:

1. Mode 1 (global reassignment) takes effect for any figure that calls `.render()`
   AFTER the reassignment — even if the figure's Plottables were constructed
   BEFORE the reassignment.
2. Mode 2 (local override) is static at the figure-module scope — the module-level
   constant is resolved when the module is first imported, and stays at that
   value for the lifetime of the process. Other modules that import
   `dsplot.style` directly still see the unmodified template default.

## Runnable demo

`research/dsplot/figures/style_override_demo.py` produces three side-by-side PNGs
that prove BOTH modes work end-to-end:

```bash
python -m research.dsplot.figures.style_override_demo
```

Outputs (in `assets/images/dsp/style_override_demo/`):

| File | Mode | What it shows |
|---|---|---|
| `default_palette.png` | baseline | template defaults (orange / purple / gold) |
| `global_override.png` | Mode 1 | global reassignment (green / pink / cyan) — affects all figures rendered after |
| `local_override.png`  | Mode 2 | local-only override (yellow / purple / orange) — global untouched |

The demo's `main()` asserts that `dsplot.style.PRIMARY_COLOR` equals its original
default after both blocks finish — proves Mode 1 was reversed in `finally` AND
Mode 2 never touched the global in the first place.
