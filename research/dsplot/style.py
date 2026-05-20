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
DEFAULT_TITLE_FONT_SIZE      = 20
DEFAULT_SUBTITLE_FONT_SIZE   = 14
DEFAULT_LABEL_FONT_SIZE      = 18
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
# below-plot subtitle isn't clipped by Figure.savefig's tight bbox.
DEFAULT_SUBTITLE_BOTTOM_PAD = 0.08
DEFAULT_TICK_LABEL_SIZE    = 18
DEFAULT_AXIS_LABEL_SIZE    = 17
# Tick mark dimensions (matplotlib defaults are ~3.5pt length, ~0.8pt width)
DEFAULT_TICK_LENGTH        = 8.0
DEFAULT_TICK_WIDTH         = 1.5
DEFAULT_SUPTITLE_FONT_SIZE = 32
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
# Change this one constant to rescale every padding in the figure.
DEFAULT_PAD_INCHES = 1.0

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
INST_FREQ_COLOR     = PRIMARY_COLOR
INST_FREQ_LINEWIDTH = 2.4
INST_FREQ_ALPHA     = 0.9
