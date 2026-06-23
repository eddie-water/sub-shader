"""dsplot colormaps — named palettes + a one-call default swap.

The single global knob for heatmap color is `style.DEFAULT_HEATMAP_CMAP`
(a matplotlib colormap *name*, resolved lazily at draw time by every
`Heatmap`). This module ships named palettes, registers them with
matplotlib, and exposes a one-call swap so changing the scheme of every
plot is a single line:

    from dsplot import colormaps
    colormaps.set_default("optichrome_cyclic")   # every heatmap, instantly

Per-plot override still wins — `Heatmap(..., cmap="optichrome_polar")` only
recolors that one field.

Optichrome palette
------------------
"Optichrome" is Felipe Pantone's saturated full-spectrum palette — the
high-chroma gradients of his kinetic op-art work. Every variant here is
anchored BLACK = lowest value, WHITE = peak, with the Optichrome hues
threading the midrange. Five takes on the same spectrum, because it reads
differently depending on how it's swept:

  optichrome        sequential black→spectrum→white (general magnitude map)
  optichrome_cyclic endpoints match (black) — wraps seamlessly; white-
                    centered peak (phase / angle / polar)
  optichrome_polar  black floor, white peak, cool→warm hue path (radial)
  optichrome_pixel  posterized discrete steps (the blocky artwork look)
  optichrome_rwbk   red/white/blue/black melt-loop (clean wrap, 2-param wheel)
  optichrome_rwbk_warm  black→maroon→red→orange→yellow→white (sequential
                        magnitude map — inferno-like, Pantone hues)
  optichrome_rwbk_cool  black→navy→…→cyan→frost→white (icy sequential mirror)

The rwbk *loop* returns to black at value 1.0, so it reads right for phase /
angle, not magnitude. The `_warm` / `_cool` cuts climb monotonically to white,
so the brightest value is the brightest color — use those for ordinary
heatmaps / spectrograms. Each sequential cut also has a posterized `_steps`
sibling. `register()` is idempotent and runs once on import.
"""
from __future__ import annotations

from typing import Sequence, Union

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from . import style

# Number of discrete bands a variant collapses to when quantized. The
# Optichrome aesthetic lives in hard-edged blocks of a small fixed palette,
# not a smooth ramp — DEFAULT_LEVELS ≈ the artwork's distinct-swatch count.
DEFAULT_LEVELS = 8

# ---------------------------------------------------------------------------
# Optichrome anchor colors — high-chroma points on the hue wheel.
# ---------------------------------------------------------------------------
OPTICHROME_BLUE    = "#1b4fd6"
OPTICHROME_CYAN    = "#22c4ec"
OPTICHROME_WHITE   = "#ffffff"
OPTICHROME_YELLOW  = "#ffd200"
OPTICHROME_ORANGE  = "#ff7a00"
OPTICHROME_RED     = "#e8231f"
OPTICHROME_CRIMSON = "#8e1437"
OPTICHROME_VIOLET  = "#4b2a8f"
OPTICHROME_BLACK   = "#0a0a0a"

# Magnitude convention (all variants): BLACK is the lowest value, WHITE is
# the peak. The saturated Optichrome hues thread the midrange so the ramp
# climbs black → cool → warm → white instead of bottoming/topping in a color.

# Sequential sweep — black floor, white peak. Cool dark hues lift off the
# floor (violet→blue), the warm spectrum carries the midbody (crimson→red→
# orange→yellow), and the brightest hue resolves into white.
_OPTICHROME_SEQUENCE: Sequence[str] = (
    OPTICHROME_BLACK,
    OPTICHROME_VIOLET,
    OPTICHROME_BLUE,
    OPTICHROME_CRIMSON,
    OPTICHROME_RED,
    OPTICHROME_ORANGE,
    OPTICHROME_YELLOW,
    OPTICHROME_WHITE,
)

# Cyclic sweep — first == last (BLACK) so the ramp tiles with no seam. WHITE
# sits dead-center as the peak, cool hues climbing into it and warm hues
# falling back out, so a wrapped/polar map still reads black-low / white-peak
# at the seam and center respectively.
_OPTICHROME_CYCLE: Sequence[str] = (
    OPTICHROME_BLACK,
    OPTICHROME_VIOLET,
    OPTICHROME_BLUE,
    OPTICHROME_CYAN,
    OPTICHROME_WHITE,
    OPTICHROME_YELLOW,
    OPTICHROME_ORANGE,
    OPTICHROME_RED,
    OPTICHROME_BLACK,
)

# Polar sweep — black floor, white peak, routed cool→warm with cyan riding
# high near the ceiling (a distinct hue path from the sequential variant so
# the two read differently on the same field).
_OPTICHROME_DIVERGING: Sequence[str] = (
    OPTICHROME_BLACK,
    OPTICHROME_VIOLET,
    OPTICHROME_BLUE,
    OPTICHROME_RED,
    OPTICHROME_ORANGE,
    OPTICHROME_YELLOW,
    OPTICHROME_CYAN,
    OPTICHROME_WHITE,
)

# Posterized steps — the literal blocky artwork palette, black floor → white
# peak, full Optichrome spectrum quantized into discrete bands.
_OPTICHROME_PIXELS: Sequence[str] = (
    OPTICHROME_BLACK,
    OPTICHROME_VIOLET,
    OPTICHROME_BLUE,
    OPTICHROME_CYAN,
    OPTICHROME_RED,
    OPTICHROME_ORANGE,
    OPTICHROME_YELLOW,
    OPTICHROME_WHITE,
)

# ---------------------------------------------------------------------------
# True Optichrome steps — the vivid block colors measured from Felipe Pantone's
# painting. Source: the clean pixel scan `optichrome_felipe_pantone.png`, whose
# native block grid (17 cols × 22 rows) was recovered by within-block-variance
# search, then every block assigned to its nearest hue family (each color below
# is that family's centroid; the weight is its measured canvas area over
# 17×22 = 374 blocks). Median-cut was rejected: it averaged across hard block
# edges into muddy tones that aren't real swatches.
#
# Thirteen swatches, not ten — finer clustering surfaced three intermediates
# the 10-family fit had folded into their neighbors: FROST (pale icy white,
# split from WHITE), OCEAN (deep blue, split from ROYAL), and NAVY (blue-black,
# split from BLACK). With those pulled out, BLACK and WHITE land truer/purer.
# Warm side maroon→red→orange→yellow, cool side frost→aqua→cyan→royal→ocean→
# violet→navy; black & white are the neutral separators (~0.13 each).
# ---------------------------------------------------------------------------
OPTI_BLACK  = "#0a0b0d"   # near-black seam / floor
OPTI_MAROON = "#62102b"   # dark maroon
OPTI_RED    = "#e01229"   # red
OPTI_ORANGE = "#f3942b"   # orange
OPTI_YELLOW = "#fccf14"   # yellow
OPTI_WHITE  = "#efefef"   # white peak
OPTI_FROST  = "#cfe1e7"   # pale icy frost (white→aqua bridge)
OPTI_AQUA   = "#9cd5e3"   # frosted aqua (pale sky blue)
OPTI_CYAN   = "#02b2d1"   # teal / sky cyan
OPTI_ROYAL  = "#046ab1"   # bright royal blue
OPTI_OCEAN  = "#014e9a"   # deep ocean blue
OPTI_VIOLET = "#423268"   # violet / indigo
OPTI_NAVY   = "#1b1c2c"   # blue-black (violet→black bridge)

# The rwbk loop in melt order, each color paired with the AREA it occupies in
# the original painting (nearest-family assignment over the 374 source blocks).
# Black & white dominate; the warm hues are thin accents — so the colormap
# spends its range proportionally, matching the canvas instead of equal bands.
# Black anchors both ends (floor + seam); white is the center peak.
_RWBK_LOOP: tuple[tuple[str, float], ...] = (
    (OPTI_BLACK,  0.136),
    (OPTI_MAROON, 0.056),
    (OPTI_RED,    0.056),
    (OPTI_ORANGE, 0.059),
    (OPTI_YELLOW, 0.059),
    (OPTI_WHITE,  0.131),
    (OPTI_FROST,  0.075),
    (OPTI_AQUA,   0.075),
    (OPTI_CYAN,   0.070),
    (OPTI_ROYAL,  0.075),
    (OPTI_OCEAN,  0.064),
    (OPTI_VIOLET, 0.064),
    (OPTI_NAVY,   0.080),
)

# Raw loop colors (equal, unweighted) — for angular uses like the 2-param
# wheel where each direction wants one distinct color.
RWBK_LOOP_COLORS: tuple[str, ...] = tuple(c for c, _ in _RWBK_LOOP)

# Sequential cuts of the rwbk loop — black floor → white peak, one half of the
# loop each. Unlike the full loop (which returns to black at value 1.0 and so
# only reads right for phase/angle), these climb monotonically to white, so
# they work as ordinary magnitude maps: the brightest value is the brightest
# color. The warm cut is inferno-like with Felipe Pantone's exact hues; the
# cool cut is the icy/electric mirror. Both make pleasant heatmaps.
_RWBK_WARM: Sequence[str] = (
    OPTI_BLACK, OPTI_MAROON, OPTI_RED, OPTI_ORANGE, OPTI_YELLOW, OPTI_WHITE,
)
_RWBK_COOL: Sequence[str] = (
    OPTI_BLACK, OPTI_NAVY, OPTI_VIOLET, OPTI_OCEAN, OPTI_ROYAL,
    OPTI_CYAN, OPTI_AQUA, OPTI_FROST, OPTI_WHITE,
)


def _weighted_listed(loop: Sequence[tuple[str, float]], total: int = 300) -> list:
    """Repeat each loop color ∝ its area weight → proportional hard bands."""
    out: list = []
    for color, weight in loop:
        out += [color] * max(1, round(weight * total))
    return out


def _proportional_nodes(loop: Sequence[tuple[str, float]]) -> list:
    """Place each color at the center of its proportional band; black anchors
    both ends so the smooth loop closes seamlessly at the seam."""
    weights = [w for _, w in loop]
    span = sum(weights)
    edges = [0.0]
    for w in weights:
        edges.append(edges[-1] + w / span)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(loop))]
    seam = loop[0][0]
    return (
        [(0.0, seam)]
        + [(centers[i], loop[i][0]) for i in range(len(loop))]
        + [(1.0, seam)]
    )


_N = 256

# Public registry: name -> Colormap. Names mirror the matplotlib registry
# so `style.DEFAULT_HEATMAP_CMAP = "optichrome_cyclic"` and direct lookups
# agree.
OPTICHROME_COLORMAPS: dict[str, Colormap] = {
    "optichrome":        LinearSegmentedColormap.from_list("optichrome", list(_OPTICHROME_SEQUENCE), N=_N),
    "optichrome_cyclic": LinearSegmentedColormap.from_list("optichrome_cyclic", list(_OPTICHROME_CYCLE), N=_N),
    "optichrome_polar":  LinearSegmentedColormap.from_list("optichrome_polar", list(_OPTICHROME_DIVERGING), N=_N),
    "optichrome_pixel":  ListedColormap(list(_OPTICHROME_PIXELS), name="optichrome_pixel"),
    "optichrome_rwbk":   LinearSegmentedColormap.from_list("optichrome_rwbk", _proportional_nodes(_RWBK_LOOP), N=_N),
    "optichrome_rwbk_steps": ListedColormap(_weighted_listed(_RWBK_LOOP), name="optichrome_rwbk_steps"),
    "optichrome_rwbk_warm": LinearSegmentedColormap.from_list("optichrome_rwbk_warm", list(_RWBK_WARM), N=_N),
    "optichrome_rwbk_warm_steps": ListedColormap(list(_RWBK_WARM), name="optichrome_rwbk_warm_steps"),
    "optichrome_rwbk_cool": LinearSegmentedColormap.from_list("optichrome_rwbk_cool", list(_RWBK_COOL), N=_N),
    "optichrome_rwbk_cool_steps": ListedColormap(list(_RWBK_COOL), name="optichrome_rwbk_cool_steps"),
}

# Stable display order — drives the showcase grid and any "list the variants"
# caller. Sequential, wrap-around, polar, posterized.
OPTICHROME_VARIANTS: tuple[str, ...] = (
    "optichrome",
    "optichrome_cyclic",
    "optichrome_polar",
    "optichrome_pixel",
)


def register() -> None:
    """Register every Optichrome colormap with matplotlib (idempotent).

    Skips names already present so a re-import or interactive re-run is a
    quiet no-op (the colormap objects are module-level singletons, so a
    name already in the registry is already ours).
    """
    for name, cmap in OPTICHROME_COLORMAPS.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap=cmap, name=name)


def get(name: str) -> Colormap:
    """Return a Colormap by name — Optichrome palettes first, else matplotlib's.

    Lets figure code pass an object straight to `Heatmap(cmap=...)` without
    caring whether the name is one of ours or a built-in.
    """
    if name in OPTICHROME_COLORMAPS:
        return OPTICHROME_COLORMAPS[name]
    return mpl.colormaps[name]


def quantize(
    cmap: Union[str, Colormap],
    n_levels: int = DEFAULT_LEVELS,
) -> ListedColormap:
    """Collapse a continuous colormap into `n_levels` hard-edged bands.

    Samples `cmap` at `n_levels` evenly spaced points and returns a
    `ListedColormap` of just those colors. Under imshow's default norm, data
    snaps to the nearest band, so the field reads as discrete blocks with
    boundaries at i/n_levels — the posterized Optichrome look — instead of a
    smooth gradient. Accepts a name (Optichrome or built-in) or a Colormap.
    """
    base = cmap if isinstance(cmap, Colormap) else get(cmap)
    colors = base(np.linspace(0.0, 1.0, n_levels))
    return ListedColormap(colors, name=f"{base.name}_q{n_levels}")


def set_default(name: str) -> str:
    """Make `name` the default colormap for every heatmap. Returns prior name.

    Registers Optichrome palettes first if needed, then repoints
    `style.DEFAULT_HEATMAP_CMAP`. Because `Heatmap` resolves that constant
    lazily at draw time, this recolors every not-yet-drawn heatmap with no
    per-plot edits. Per-plot `cmap=` overrides still take precedence.
    """
    if name in OPTICHROME_COLORMAPS and name not in mpl.colormaps:
        register()
    previous = style.DEFAULT_HEATMAP_CMAP
    style.DEFAULT_HEATMAP_CMAP = name
    return previous


# Register on import so a bare `cmap="optichrome_cyclic"` resolves anywhere
# dsplot is imported, without an explicit register() call.
register()
