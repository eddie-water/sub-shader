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
GRID_WAVEFORM_COLOR = "#fcfeaa"
FREQ_LINE_COLOR = "#AAAAAA"
SUBTITLE_COLOR = "black"

# ===== FONT SIZES =====
TITLE_FONT_SIZE = 24
TICK_LABEL_SIZE = 20
AXIS_LABEL_FONT_SIZE = 24
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
LEFT_MARGIN = 0.1
RIGHT_MARGIN = 0.9
BOTTOM_MARGIN = 0.05
TOP_MARGIN = 0.9
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
STUB_DPI = 100

# ===== ATOMIC PLOTTERS =====
# Visual constants for the atomic plotters in plotting.py
TICK_LABEL_COLOR = "#888888"
SPINE_COLOR = "#444444"
SPINE_LINEWIDTH = 0.8
AXIS_GRID_COLOR = "white"
AXIS_GRID_ALPHA = 0.08
AXIS_GRID_LINEWIDTH = 0.5
WAVEFORM_YLIM_PADDING = 1.1
INST_FREQ_LINEWIDTH = 1.5
INST_FREQ_ALPHA = 0.9
FFT_LINEWIDTH = 1.0
VMAX_PERCENTILE = 99.0

# ===== DSP.md FOUNDATION FIGURES — VECTOR PLOTS =====
# Primitives for the §2.4 vector figures (vector_basics, dot_product_geometry,
# vector_similarity, vector_projection). All knobs in one place so the look
# stays consistent across panels.

# Vector colors — `a` is the reference vector; `b` is the one being compared
# to / projected onto a. PROJECTION is b's shadow on a (the dot-product
# magnitude visualized).
VECTOR_A_COLOR = "#fcfeaa"      # warm yellow — matches GRID_WAVEFORM_COLOR
VECTOR_B_COLOR = "#7ec8ff"      # cool blue — distinct from a
VECTOR_PROJ_COLOR = "#ff8c66"   # orange — projection / "shadow"
VECTOR_NEUTRAL_COLOR = "#888888"  # for misc / sample vectors with no role
VECTOR_AXIS_COLOR = "#444444"
VECTOR_AXIS_ALPHA = 0.6
VECTOR_DROPLINE_COLOR = "#888888"   # dashed perpendicular from b to projection
VECTOR_DROPLINE_ALPHA = 0.55

VECTOR_LINEWIDTH = 2.4
VECTOR_HEAD_WIDTH = 0.10        # in data units, axes are typically [-1, 1]
VECTOR_HEAD_LENGTH = 0.14
VECTOR_LABEL_FONT_SIZE = 18
VECTOR_LABEL_OFFSET = 0.06      # axis-units offset from arrow tip
VECTOR_PANEL_TITLE_SIZE = 18
VECTOR_PANEL_RESULT_SIZE = 16   # for the "= +1.0" / "= 0" annotations
VECTOR_PANEL_RESULT_COLOR = "#dddddd"
VECTOR_DEFAULT_LIM = 1.25       # symmetric ±lim axis range
VECTOR_FIGSIZE_PER_PANEL = 5.0  # square panel width in inches
VECTOR_FIGSIZE_HEIGHT = 5.0

# ===== DSP.md MOTIVATOR FIGURE =====
# Compact 3-row layout for the §1 motivator. All "MOTIVATOR_*" knobs in one
# place so the figure can be retuned without hunting through dsp_figures.py.
MOTIVATOR_TICK_LABEL_SIZE = 14
MOTIVATOR_AXIS_LABEL_SIZE = 14
MOTIVATOR_ROW_LABEL_SIZE = 16
# Fraction of figure HEIGHT used as a uniform margin on all four sides
# (horizontal margin is scaled by H/W so all edges are equal in inches).
MOTIVATOR_MARGIN = 0.05
# Fraction of axes height used as inter-panel gap. Picked so the gap roughly
# equals the edge margin in absolute inches at the default figsize.
MOTIVATOR_HSPACE = 0.18
# Width-ratio of label column to data column in the 2-col gridspec.
# 0.40 → label col gets ~29% of figure width.
MOTIVATOR_LABEL_RATIO = 0.18
# Hard-clamp the visible time axis to this value (in seconds). The chirp's
# 20 kHz endpoint is timed to land here so the curve crests at the right edge.
MOTIVATOR_VISIBLE_END_S = 1.0
