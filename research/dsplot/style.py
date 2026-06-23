"""Inheritable style template for dsplot — D-05.

Every dsplot figure inherits these defaults. Two override modes:

  (1) GLOBAL reassignment — affects every figure rendered after:
          import dsplot
          dsplot.style.PRIMARY_COLOR = "#new"

  (2) LOCAL override — a figure module derives its own constant from a default,
      leaving the global untouched (affects only that figure):
          from dsplot import style
          LABEL_RATIO = style.DEFAULT_LABEL_RATIO * 1.5

Plottables resolve None-valued style knobs against dsplot.style.* at draw()
time (lazy lookup), so a global reassignment between construction and draw
uses the NEW value. Local overrides in figure modules just shadow at the
call site.

Naming convention:
  - Palette colors use ROLE names directly (PRIMARY_COLOR, BG_COLOR,
    DROPLINE_COLOR) — they identify a role, not a default.
  - Everything else uses DEFAULT_* prefix — they're the inheritable defaults
    figures can override locally.
"""

# ============================================================
# PALETTE (role-named — identity slots, not "defaults")
# ============================================================
PRIMARY_COLOR    = "#ff5a1f"
SECONDARY_COLOR  = "#7b6fe1"
TERTIARY_COLOR   = "#ffd27d"
NEUTRAL_COLOR    = "#EEEEEE"
HIGHLIGHT_COLOR  = "#22d3ee"
BG_COLOR         = "#1A1A1A"
SPINE_COLOR      = "#444444"
TICK_LABEL_COLOR = "#888888"
DROPLINE_COLOR   = "#888888"

# ============================================================
# LINEWORK
# ============================================================
DEFAULT_VECTOR_LINEWIDTH      = 2.4
DEFAULT_VECTOR_BOLD_LINEWIDTH = 3.6
DEFAULT_SPINE_LINEWIDTH       = 0.8
DEFAULT_DROPLINE_LINEWIDTH    = 1.2
DEFAULT_DROPLINE_ALPHA        = 0.55
DEFAULT_DROPLINE_LINESTYLE    = "--"

# ============================================================
# ARROWHEADS
# ============================================================
DEFAULT_ARROW_HEAD_LENGTH = 0.3
DEFAULT_ARROW_HEAD_WIDTH  = 0.1
DEFAULT_ARROW_MUTATION    = 22

# ============================================================
# TYPOGRAPHY
# ============================================================
DEFAULT_TITLE_FONT_SIZE      = 32
DEFAULT_SUBTITLE_FONT_SIZE   = 26
DEFAULT_LABEL_FONT_SIZE      = 26
# Decorator/callout text inside a panel (e.g. "peak", "centroid" labels
# that annotate a feature). Distinct from DEFAULT_LABEL_FONT_SIZE, which
# is for in-figure math labels like "aₓ" / "aᵧ" component labels.
DEFAULT_ANNOTATION_FONT_SIZE = 11
# Y position (axes coords) for the subtitle. Negative = below the plot box.
# Panels with a subtitle declare a bottom_pad so the figure reserves room.
DEFAULT_SUBTITLE_Y         = -0.12
# Y position (axes coords) for the title (a hair above the plot box).
DEFAULT_TITLE_Y            = 1.04
# Extra figure-bottom pad reserved when any panel has a subtitle, so the
# below-plot subtitle isn't clipped by Figure.savefig's tight bbox. Bumped
# alongside the subtitle font size (now matches title at 20pt).
DEFAULT_SUBTITLE_BOTTOM_PAD = 0.10
# Caption sits below the subtitle. Italic + bold matches the prior subtitle
# styling; size 14 keeps it visibly smaller than the bumped subtitle (now 20).
DEFAULT_CAPTION_FONT_SIZE  = 30
DEFAULT_CAPTION_Y          = -0.26
# Figure-level caption chrome — `Figure N` identifier + explanatory caption
# rendered at the bottom of the figure (vs. panel-level subtitle/caption).
DEFAULT_FIGURE_NUMBER_FONT_SIZE  = 26
DEFAULT_FIGURE_CAPTION_FONT_SIZE = 26
# Extra figure-bottom pad reserved when any panel has a caption, on top of
# whatever the subtitle reserves.
DEFAULT_CAPTION_BOTTOM_PAD = 0.14
# Figure 1 (the cross-figure style guide) sets the shared standard: axis labels
# render LARGER than tick numbers so the unit label is the dominant voice, and
# captions (DEFAULT_CAPTION_FONT_SIZE) match the tick-number size so axis chrome
# and body text read as one type system.
DEFAULT_TICK_LABEL_SIZE    = 30
DEFAULT_AXIS_LABEL_SIZE    = 36
# Tick mark dimensions (matplotlib defaults are ~3.5pt length, ~0.8pt width)
DEFAULT_TICK_LENGTH        = 8.0
DEFAULT_TICK_WIDTH         = 1.5
# Inch-domain offset from the axes spine to the axis-label text. Constrained
# by a hard upper bound — the y-label MUST fit inside half the inter-cell
# gutter so the rotated text doesn't visually encroach into the neighboring
# cell's chrome territory. With DEFAULT_GUTTER = 2.0" → half-gutter = 1.0";
# y-inset = 0.7" leaves 0.3" between the label center and the cell border
# (well clear after the rotated-text half-width is accounted for). x-tick
# labels project vertically only (constrained by font height ~0.3" at 22pt),
# so x-inset can be tighter than y-inset and still clear the tick labels.
# DEFAULT_AXIS_LABEL_INSET_INCHES retained for back-compat (= the x-default).
# Axis label center sits at PAD/2 inches from the spine — exactly half-way
# in the padding zone between the spine and the cell border (PAD = 1.5 →
# inset = 0.75). Hardcoded rather than derived to keep style.py top-down
# readable (DEFAULT_PAD_INCHES is declared below in the LAYOUT section).
DEFAULT_X_AXIS_LABEL_INSET_INCHES = 0.75
DEFAULT_Y_AXIS_LABEL_INSET_INCHES = 0.75
DEFAULT_AXIS_LABEL_INSET_INCHES = DEFAULT_X_AXIS_LABEL_INSET_INCHES
DEFAULT_SUPTITLE_FONT_SIZE = 40
# SUPTITLE_* family — sibling constants for SuptitlePanel (mirrors TITLE_*
# pattern). Defaults preserve the legacy `_mpl_fig.suptitle(...)` rendering
# contract bit-identically; `DEFAULT_SUPTITLE_FONT_SIZE` is left intact
# because `figure.py` still reads it in the legacy sugar path.
SUPTITLE_FONT_SIZE = DEFAULT_SUPTITLE_FONT_SIZE  # 32
SUPTITLE_COLOR     = TICK_LABEL_COLOR             # "#888888"
SUPTITLE_WEIGHT    = "bold"
DEFAULT_ROW_LABEL_SIZE     = 16

# ============================================================
# PANEL SIZING + DPI
# ============================================================
DEFAULT_DPI               = 150
DEFAULT_PANEL_SIZE_INCHES = 5.0
DEFAULT_PANEL_MARGIN      = 0.05
DEFAULT_PANEL_UNIT_INCHES = 4.0

# ============================================================
# ROW / GRID LAYOUT
# ============================================================
DEFAULT_HSPACE      = 0.18
DEFAULT_WSPACE      = 0.04
DEFAULT_LABEL_RATIO = 0.18

# THE unitary spacing knob — every cell in the figure has this much padding
# on all 4 sides (between its cell border and its inner panel spine). Cells
# tile densely:
#   - perimeter cells touch the figure edge → perimeter margin = 1 PAD
#   - adjacent cells share a border at the midpoint of the inter-cell gutter,
#     so inter-cell gutter = 2 PAD (each adjacent cell contributes 1 PAD)
# Change this one constant to rescale every padding in the figure. Axis
# labels live inside this padding band (their inset from the spine is set
# by DEFAULT_{X,Y}_AXIS_LABEL_INSET_INCHES); PAD must be > max(inset) so the
# label has visible breathing room between itself and the cell border.
DEFAULT_PAD_INCHES = 1.5

# Derived: perimeter cell's outer padding = 1 PAD = the figure margin.
DEFAULT_MARGIN_INCHES = DEFAULT_PAD_INCHES
# Derived: inter-cell gutter = two cells × 1 PAD per cell = 2 PAD.
# Rows and columns share the same gutter unit — uniform spacing.
DEFAULT_GUTTER_INCHES = 2.0 * DEFAULT_PAD_INCHES
DEFAULT_COLUMN_GUTTER_INCHES = 2.0 * DEFAULT_PAD_INCHES
# Reserved title band sitting above the axes spine. Title text V-centers
# inside this band, so a larger value pushes the title higher above the
# spine and gives more breathing room between title and content.
DEFAULT_PANEL_TITLE_RESERVE_INCHES = 0.8
# Legacy constant retained for any caller that still reads it; not used by
# the current Figure.compose layout math.
DEFAULT_AXIS_LABEL_RESERVE_INCHES = 0.75
# Content border for text-only panels (TextPanel / DynamicTextPanel): the box
# extends OUT to the cell border (by DEFAULT_PAD_INCHES per side) then insets by
# this fraction of the cell width. Smaller fraction => bigger box, closer to the
# cell edge. Shared so every text panel that shows a border gets the same
# generous "box just inside the cell" look (see figure 1's row labels).
DEFAULT_CONTENT_BORDER_INSET_FRAC = 1.0 / 16.0
# Inner gutter for CompositePanel's nested gridspec. Composite inner cells are
# physically smaller than top-level cells (they share one outer cell), so the
# outer gutter would eat too much of the inner real estate. 0.6" is enough to
# host axis decoration on tightly-stacked inner panels.
DEFAULT_INNER_GUTTER_INCHES = 0.6

# ============================================================
# VECTOR-AXES DECORATION
# ============================================================
DEFAULT_VECTOR_LABEL_OFFSET = 0.30
DEFAULT_VECTOR_LIM          = 4.0
# Alpha applied to a plottable that's been demoted to "framing" — present for
# context but not the subject of the panel. Used when a figure spotlights an
# overlay (components, projection) ON TOP of muted base vectors.
DEFAULT_MUTED_ALPHA         = 0.6
DEFAULT_AXIS_GRID_COLOR     = "white"
DEFAULT_AXIS_GRID_ALPHA     = 0.08
DEFAULT_AXIS_GRID_LINEWIDTH = 0.5
DEFAULT_AXIS_ARROW_INSET    = 0.05
DEFAULT_AXIS_LABEL_OFFSET   = 0.08

# ============================================================
# HEATMAP
# ============================================================
DEFAULT_HEATMAP_CMAP            = "inferno"
DEFAULT_HEATMAP_VMAX_PERCENTILE = 99.0

# ============================================================
# INST FREQ OVERLAY
# ============================================================
INST_FREQ_COLOR     = TERTIARY_COLOR
INST_FREQ_LINEWIDTH = 3.0
INST_FREQ_ALPHA     = 1.0


# ============================================================
# NOTEBOOK COMPACT PROFILE
# ============================================================
# Shared nb-display profile. Style constants above are sized for production
# print at ~unit_inches=4.0; at nb scale (~unit_inches=2.0–2.5) the chrome
# inches and font points dominate cells. Every figure module's show()/embed()
# wraps build + render in `with nb_compact_style(): ...` so the entire dsplot
# library shares one nb visual tone — figures stay consistent.
#
# To rebalance the nb tone, edit the values in NB_COMPACT_OVERRIDES below;
# the change applies to every figure's show() simultaneously.
from contextlib import contextmanager as _contextmanager
import sys as _sys

NB_COMPACT_OVERRIDES = {
    # Fonts — sized for ~2.0–2.5" cells.
    "DEFAULT_TITLE_FONT_SIZE": 16,
    "DEFAULT_SUBTITLE_FONT_SIZE": 16,
    "DEFAULT_LABEL_FONT_SIZE": 16,
    "DEFAULT_CAPTION_FONT_SIZE": 12,
    "DEFAULT_TICK_LABEL_SIZE": 12,
    "DEFAULT_AXIS_LABEL_SIZE": 16,
    "DEFAULT_SUPTITLE_FONT_SIZE": 16,
    "SUPTITLE_FONT_SIZE": 20,
    "DEFAULT_FIGURE_NUMBER_FONT_SIZE": 13,
    "DEFAULT_FIGURE_CAPTION_FONT_SIZE": 10,
    "DEFAULT_TICK_LENGTH": 4.0,
    "DEFAULT_TICK_WIDTH": 1.0,
    # Layout inches — chrome zone shrinks to leave room for the plot inside
    # smaller cells. PAD bumped 0.5 → 0.7 vs prior pass so axis labels have
    # 0.40" (not 0.20") between themselves and the cell border.
    "DEFAULT_PAD_INCHES": 0.7,
    "DEFAULT_MARGIN_INCHES": 0.7,
    "DEFAULT_GUTTER_INCHES": 1.4,
    "DEFAULT_COLUMN_GUTTER_INCHES": 1.4,
    "DEFAULT_INNER_GUTTER_INCHES": 0.3,
    "DEFAULT_PANEL_TITLE_RESERVE_INCHES": 0.4,
    # Axis label inset = PAD/2 = half_gutter/2 = midpoint between spine
    # and cell border. Centers the label between them.
    "DEFAULT_X_AXIS_LABEL_INSET_INCHES": 0.35,
    "DEFAULT_Y_AXIS_LABEL_INSET_INCHES": 0.35,
    "DEFAULT_AXIS_LABEL_INSET_INCHES": 0.35,
    # Plottable-specific (cosmetic — keeps lines from looking heavy in nb).
    "INST_FREQ_LINEWIDTH": 1.4,
}


@_contextmanager
def nb_compact_style():
    """Temporarily apply NB_COMPACT_OVERRIDES to this module.

    Lazy lookup in plottables resolves `dsplot.style.*` at draw() time, so
    every figure rendered inside this context picks up the compact values.
    Originals are restored on exit — production `render()` paths outside
    the context window keep their print-scale chrome.
    """
    _module = _sys.modules[__name__]
    _orig = {k: getattr(_module, k) for k in NB_COMPACT_OVERRIDES}
    try:
        for k, v in NB_COMPACT_OVERRIDES.items():
            setattr(_module, k, v)
        yield
    finally:
        for k, v in _orig.items():
            setattr(_module, k, v)
