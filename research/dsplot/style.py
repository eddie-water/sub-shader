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
PRIMARY_COLOR    = "#e1641a"
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
DEFAULT_TITLE_FONT_SIZE      = 16
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
DEFAULT_TICK_LABEL_SIZE    = 14
DEFAULT_AXIS_LABEL_SIZE    = 14
DEFAULT_SUPTITLE_FONT_SIZE = 32
DEFAULT_ROW_LABEL_SIZE     = 16

# ============================================================
# PANEL SIZING + DPI
# ============================================================
DEFAULT_DPI               = 150
DEFAULT_PANEL_SIZE_INCHES = 5.0
DEFAULT_PANEL_MARGIN      = 0.05

# ============================================================
# ROW / GRID LAYOUT
# ============================================================
DEFAULT_HSPACE      = 0.18
DEFAULT_WSPACE      = 0.04
DEFAULT_LABEL_RATIO = 0.18

# Outer figure margins (subplots_adjust knobs). These define the gutter
# between the panel-grid extent and the figure perimeter — analogous to
# CSS margins. Since `Figure.savefig()` saves at exact `figsize × dpi`,
# these margins are the SOLE source of "breathing room" around the
# panel grid. Decorators anchored outside this gutter will clip.
DEFAULT_MARGIN_LEFT   = 0.06
DEFAULT_MARGIN_RIGHT  = 0.94
DEFAULT_MARGIN_BOTTOM = 0.07
# DEFAULT_MARGIN_TOP is computed by Figure.render() from suptitle presence
# (top_pad reserves room for the suptitle when one is set).

# ============================================================
# VECTOR-AXES DECORATION
# ============================================================
DEFAULT_VECTOR_LABEL_OFFSET = 0.30
DEFAULT_VECTOR_LIM          = 1.25
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
