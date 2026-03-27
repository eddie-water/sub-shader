"""
Canonical visual style constants for all SubShader figures.

Single source of truth — every color, fontsize, linewidth, alpha, figsize,
and spacing value lives here. Import as:
    from utilities import style  (from research/ CWD)
    or: from research.utilities import style
"""

# ===== COLORS =====
BG_COLOR = "#1A1A1A"
WAVEFORM_COLOR = "#606060"
GRID_WAVEFORM_COLOR = "#ffffcf"
FREQ_LINE_COLOR = "#AAAAAA"
SUBTITLE_COLOR = "black"

# ===== FONT SIZES =====
TITLE_FONT_SIZE = 24
TICK_LABEL_SIZE = 14
AXIS_LABEL_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 32
SUBTITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 24

# ===== LINE WIDTHS =====
FREQ_LINE_WIDTH = 2
WAVEFORM_ALPHA = 0.75

# ===== FIGURE DIMENSIONS =====
FIGURE_WIDTH = 20
ROW_HEIGHT = 4

# ===== LAYOUT SPACING =====
HSPACE = 0.22
LEFT_MARGIN = 0.06
RIGHT_MARGIN = 0.94
BOTTOM_MARGIN = 0.04
TOP_MARGIN = 0.90
SUPTITLE_Y = 0.975
SUBTITLE_Y = 0.925

# ===== COMPARISON GRID =====
GRID_CMAP = "inferno"
GRID_HSPACE = 0.08
GRID_WSPACE = 0.04
GRID_MARGIN = 0.05
GRID_FIGSIZE_W = 24
GRID_FIGSIZE_H = 16
GRID_TITLE_PAD = 20
LABEL_CHAR_WIDTH = 0.028
LABEL_PAD = 0.14

# ===== RENDERING =====
DEFAULT_DPI = 150
