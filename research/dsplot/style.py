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
# Weights below are tuned for the common 28" canvas (style.FIGURE_WIDTH_INCHES).
# They were ~2x thinner when figures rendered at ~13-16"; on the wide canvas an
# absolute-pt line reads ~half as thick, so the vector family is scaled up to
# stay bold. These constants are vector-panel-only (fig 1 has no vectors), so
# the bump is isolated from the spectrogram figures.
# ONE hero linework weight shared by vectors, stem plots, parallel/dashed
# components, and accumulator stems across every figure — so a stem in 2.5
# reads at the same thickness as a vector in 2.4.1 / 2.4.2 and a component arm
# in 2.4.3. Both the "thin" and "bold" vector weights are this single value
# (no thin/bold split anymore) so component arrows and main vectors match.
# Bumped 9.0 → 11.0 → 12.5 → 16.0 for a heavier, more confident line across the
# family (paired with smaller arrowheads below so the heads don't bulk up).
DEFAULT_VECTOR_LINEWIDTH      = 16.0
DEFAULT_VECTOR_BOLD_LINEWIDTH = 16.0
DEFAULT_SPINE_LINEWIDTH       = 0.8
# 3D axis spines are manual data-space Vector lines (static_panel_3d.draw_3d_spines),
# NOT matplotlib spines — so DEFAULT_SPINE_LINEWIDTH (a 2D-panel knob) doesn't reach
# them. They get their own weight here: quiet chrome, well under the 12.5 vector line
# they used to borrow, so the 3D vectors read as the subject and the spines recede.
DEFAULT_SPINE_3D_LINEWIDTH    = 4.0
DEFAULT_DROPLINE_LINEWIDTH    = 2.6
DEFAULT_DROPLINE_ALPHA        = 0.55
DEFAULT_DROPLINE_LINESTYLE    = "--"

# ============================================================
# PANEL FRAME (the single visible box around every plot)
# ============================================================
# ONE frame model across every figure (the figure-1 / 2.5 model): per-axes
# spines are off, and the cell border drawn by Figure(show_cell_borders=True)
# IS each panel's frame. These knobs make that single frame a clean, visible
# thin line shared everywhere — change them once, every figure's frame updates.
DEFAULT_FRAME_COLOR     = NEUTRAL_COLOR   # "#EEEEEE"
# ONE common cell-border weight for EVERY figure — matches figure 1's border
# (4.5pt, the thickness the user signed off on). The earlier per-figure
# width-scaling (3.0 base, 2.5→4.14, 2.6→4.62) was abandoned: the montage tiles
# all figures at a common px/inch, so equal POINTS read as equal pixels — a
# single shared value is what actually makes the borders look identical there.
DEFAULT_FRAME_LINEWIDTH = 4.5
DEFAULT_FRAME_ALPHA     = 1.0

# ============================================================
# COMMON CANVAS — cross-figure type consistency
# ============================================================
# Every figure renders at ONE physical width + dpi so a shared point-size type
# scale (header / label / annotation / body-text / tick) reads at the SAME
# visual size in every figure, both in the montage and in DSP.md. 28" @ 150dpi
# is figure 1's canvas — the cross-figure gold standard — so conforming the
# other figures to it leaves fig 1 untouched as the reference. Compose figures
# opt in via `Figure.compose(total_width_inches=style.FIGURE_WIDTH_INCHES)`,
# which back-solves unit_inches to hit this width.
FIGURE_WIDTH_INCHES = 28.0
FIGURE_DPI          = 150

# Shared canvas WIDTH (inches) for the figure 2.5 / 2.6 PAIR, so the two sit as a
# matched pair in DSP.md. They share this width (via
# `Figure.compose(total_width_inches=...)`) AND a common per-row cell height
# (both pass `unit_height_inches=SHARED_UNIT_INCHES`), so a single row of 2.6 has
# the SAME vertical proportions as a single row of 2.5 — graph-paper consistency.
# They do NOT share total height: 2.5 has 3 case rows, 2.6 has 2, so 2.6 is
# naturally shorter. Value = 2.6's natural width (the wider of the two), so
# matching widens 2.5 slightly rather than shrinking 2.6.
PAIR_25_26_WIDTH_INCHES  = 43.1

# Fixed PHYSICAL height (inches) of the top header band on the common 28"
# canvas, so every figure's title sits in an identically-sized band with the
# same air above/below — regardless of its column count. Without this the band
# height is `row_heights[0] × unit_height`, and unit_height swings ~3× between a
# 3-column and a 9-column figure, so the same relative row-height renders a wildly
# different physical band (titles float in one figure, cram against the panels in
# another). 2.0" matches fig-1's gold-standard band (STACK_BAND_HEIGHT 0.4 ×
# STACK_UNIT_INCHES 5.0 = 2.0"), so every figure's title sits in a band the same
# physical height as fig 1's, with the 44pt title V-centered the same way.
# Opt in via `Figure.compose(header_band_inches=style.HEADER_BAND_INCHES)`.
HEADER_BAND_INCHES = 2.0

# Shared physical size (inches) of ONE tile — the cross-figure "graph paper"
# unit that §2.4.1-onward compose on (pass
# `Figure.compose(unit_inches=style.SHARED_UNIT_INCHES)`, DO NOT pass
# total_width_inches), so a 1×1 cell is the same size in every figure and a
# panel that spans N×M cells matches its twin in another figure exactly.
# Figures end up at DIFFERENT widths (width = n_cols·unit + gutters + margins) —
# the montage tiles them at true scale instead of squashing every figure to one
# width. Value = figure 1's TEXT BOX (its caption square = 2×STACK_SQUARE_INCHES
# = 10.0"): the user picked fig-1's caption box as THE canonical tile, so a
# 1-unit plot here equals fig-1's caption box exactly. Fig 1 keeps its own finer
# 5" base grid internally (untouched reference); its text box == this tile.
SHARED_UNIT_INCHES = 10.0


def unit_inches_for_width(n_cols, total_width_inches=None):
    """Back-solve the per-column unit width that makes an n_cols compose figure
    exactly `total_width_inches` (default FIGURE_WIDTH_INCHES) wide. Width is
    n_cols·unit + (n_cols-1)·col_gutter + 2·margin, so unit = (W - 2·margin -
    (n_cols-1)·col_gutter) / n_cols. Used by compose(total_width_inches=...) and
    by figures that need the resolved unit width up front (e.g. square-cell math).
    Resolves margin/gutter at call time so a style override is honored."""
    W = total_width_inches if total_width_inches is not None else FIGURE_WIDTH_INCHES
    return (W - 2.0 * DEFAULT_MARGIN_INCHES
            - (n_cols - 1) * DEFAULT_COLUMN_GUTTER_INCHES) / n_cols
# Breathing room (inches) between the figure's outer frame and the PNG edge.
# A fixed inch value gives every figure the SAME visible gap regardless of its
# pixel size / dpi. The library draws the outer perimeter inset by this much on
# all four sides so the frame reads as "a framed figure on a page" rather than
# a box jammed against (and half-clipped by) the canvas edge. One knob → every
# figure's edge gap moves together.
DEFAULT_FRAME_EDGE_GAP_INCHES = 0.05

# ============================================================
# ARROWHEADS
# ============================================================
# Arrowhead geometry (multiplied by DEFAULT_ARROW_MUTATION → points). A long,
# NARROW head: the base is kept slim (the wide-base look read as cluttered when
# many arrows sit together in 242's staircase) while the length stays up so the
# head still reads as a distinct triangle, not a stubby flare. Mutation scales
# the whole head uniformly so the L/W proportion is preserved. At M26: head_width
# 0.36 × 26 ≈ 9.4pt base — slim but still flaring past the 16pt shaft once the
# FancyArrowPatch head renders wider than its point count; head_length 0.74 × 26
# ≈ 19.2pt keeps a clean, pointed tip.
DEFAULT_ARROW_HEAD_LENGTH = 0.74
DEFAULT_ARROW_HEAD_WIDTH  = 0.36
DEFAULT_ARROW_MUTATION    = 26

# ============================================================
# TYPOGRAPHY
# ============================================================
DEFAULT_TITLE_FONT_SIZE      = 44
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
DEFAULT_CAPTION_FONT_SIZE  = 40
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
DEFAULT_TICK_LABEL_SIZE    = 40
DEFAULT_AXIS_LABEL_SIZE    = 46
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
DEFAULT_SUPTITLE_FONT_SIZE = 44
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
# Light but actually VISIBLE at montage / phone scale: 0.08 rendered at full res
# but vanished once the montage downscaled it. 0.16 + a 0.9 line reads as a quiet
# grid without competing with the vectors.
DEFAULT_AXIS_GRID_ALPHA     = 0.16
# 3D floor (z=0 plane) grid sits a touch higher than the 2D grid — perspective
# foreshortening + the dark scene dim it further.
DEFAULT_AXIS_GRID_ALPHA_3D  = 0.22
DEFAULT_AXIS_GRID_LINEWIDTH = 0.9
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
# CHROME EMPHASIS (font weight / style for chrome text)
# ============================================================
# Titles render bold; subtitles and captions render bold-italic. Extracted from
# panel/figure draw code so the whole library shares one emphasis vocabulary.
DEFAULT_TITLE_FONT_WEIGHT      = "bold"
DEFAULT_SUBTITLE_FONT_STYLE    = "italic"
DEFAULT_SUBTITLE_FONT_WEIGHT   = "bold"
DEFAULT_CAPTION_FONT_STYLE     = "italic"
DEFAULT_CAPTION_FONT_WEIGHT    = "bold"
# Figure-level chrome (the "Figure N" identifier + its caption) shares the
# subtitle/caption bold-italic emphasis.
DEFAULT_FIGURE_NUMBER_FONT_WEIGHT  = "bold"
DEFAULT_FIGURE_NUMBER_FONT_STYLE   = "italic"
DEFAULT_FIGURE_CAPTION_FONT_WEIGHT = "bold"
DEFAULT_FIGURE_CAPTION_FONT_STYLE  = "italic"
# Monospace family for numeric readouts (e.g. dot-product result text).
DEFAULT_MONO_FONT_FAMILY = "monospace"
# THE sans family for ALL non-numeric text (titles, axis/tick labels, body
# copy, glyphs) across every figure. Set into rcParams at Figure creation so
# every text artist inherits it — the single knob for the figure type system.
# Numeric readouts opt into DEFAULT_MONO_FONT_FAMILY explicitly and override it.
DEFAULT_FONT_FAMILY = "DejaVu Sans"

# ============================================================
# TICK DECORATION (direction + inset scaling)
# ============================================================
# Data panels draw ticks that point INWARD and are scaled DOWN from the base
# tick dims (DEFAULT_TICK_LENGTH / DEFAULT_TICK_WIDTH) so the cell border, not
# the ticks, reads as the panel frame. Scaling (not absolute sizes) keeps the
# relationship intact when a render profile rescales the base dims.
DEFAULT_TICK_DIRECTION          = "inout"
DEFAULT_INSET_TICK_LENGTH_SCALE = 0.6
DEFAULT_INSET_TICK_WIDTH_SCALE  = 0.8
# Heatmaps point ticks AWAY from the data so they don't overlay the image.
DEFAULT_HEATMAP_TICK_DIRECTION  = "out"
# Heatmap axis inset from the cell edge, inches.
DEFAULT_HEATMAP_AXIS_EDGE_CLEARANCE_INCHES = 0.15

# ============================================================
# VECTOR-AXES DECORATION — weights, label glyphs, readout
# (companion to the VECTOR-AXES DECORATION block above)
# ============================================================
# Faint x–y reference-axis (crosshair / arrow) weight a vector sits on.
DEFAULT_AXIS_DECORATION_ALPHA     = 0.85
# 28"-canvas weights (were 1.8 / 0.9 / 14 at ~13"): the faint x-y reference
# axis the vectors sit on must stay visible at the wider canvas — but thin enough
# to read as a quiet reference crosshair, not a structural line competing with the
# vectors (the grid carries the scale).
DEFAULT_AXIS_DECORATION_LINEWIDTH = 2.0
DEFAULT_AXIS_ARROW_MIN_LINEWIDTH  = 1.8
DEFAULT_AXIS_ARROW_MUTATION       = 28
# x/y glyph labels ("x", "y") sit at this fraction of the axis limit, render a
# touch larger than tick numbers, and carry the axis emphasis. Pushed back OUT
# toward the spine tip (0.84 → 0.96) so x/y ride near the cell border instead of
# floating inside it — the labels read as "this edge IS the axis".
DEFAULT_AXIS_LABEL_POS_FRAC    = 0.96
DEFAULT_AXIS_LABEL_SIZE_BUMP   = 2
DEFAULT_AXIS_LABEL_ALPHA       = 0.95
DEFAULT_AXIS_LABEL_FONT_WEIGHT = "bold"
DEFAULT_AXIS_LABEL_FONT_STYLE  = "italic"
# Numeric result readout below a vector panel (axes-fraction y).
DEFAULT_RESULT_TEXT_Y = -0.08

# ============================================================
# PLOTTABLE MARKERS
# ============================================================
DEFAULT_STEM_MARKER          = "o"
DEFAULT_STEM_MARKERSIZE      = 6.0
# The ONE circle size for every hero stem tip across figures (data stems,
# spectrum coefficient stems, accumulator/sum bars). Stems are differentiated by
# HEIGHT alone — the tip dome is uniform everywhere — so a tall stem may run off
# the panel ("half plotted") rather than shrinking its marker to fit. Domes the
# 16.0 hero stem at a ~1.6 marker:stem ratio.
DEFAULT_HERO_STEM_MARKERSIZE = 26.0
DEFAULT_SPOTLIGHT_SIZE       = 100
DEFAULT_SPOTLIGHT_EDGE_COLOR = "white"
# 3D vector tip-marker size + the extra linewidth a 3D bold vector gets over 2D.
DEFAULT_VECTOR_3D_TIP_SIZE            = 80
DEFAULT_VECTOR_3D_BOLD_LINEWIDTH_BUMP = 0.6

# ============================================================
# ACCUMULATOR STRIP (running-total / dot-product bar)
# ============================================================
# A tall narrow strip carrying ONE heavy stem from a zero baseline up/down to a
# value, with the value printed big on the empty side. Lifted from figure 2.5's
# Sum strip so 2.4.2 / 2.4.3 / 2.5 / 2.6 share one accumulator vocabulary.
# Accumulator stem shares the hero linework weight so the running-total bar
# reads at the same thickness as the data stems / vectors beside it (figure
# 2.5's look). Tracks DEFAULT_VECTOR_LINEWIDTH (12.5 → 16.0).
DEFAULT_ACCUM_STEM_LINEWIDTH    = 16.0
DEFAULT_ACCUM_MARKERSIZE        = DEFAULT_HERO_STEM_MARKERSIZE
DEFAULT_ACCUM_ZERO_LINE_WIDTH   = 1.8
DEFAULT_ACCUM_ZERO_LINE_ALPHA   = 0.6
DEFAULT_ACCUM_READOUT_FONT_SIZE = 22
# Stem/readout color (sign read by direction off the zero line); readout uses
# the chrome text color so it reads as part of the type system.
DEFAULT_ACCUM_STEM_COLOR    = NEUTRAL_COLOR
DEFAULT_ACCUM_READOUT_COLOR = NEUTRAL_COLOR

# ============================================================
# TEXT PANEL
# ============================================================
DEFAULT_TEXT_FONT_WEIGHT       = "bold"
DEFAULT_TEXT_MIN_FONT_SIZE     = 14.0
DEFAULT_TEXT_CELL_PADDING_FRAC = 0.08
DEFAULT_TEXT_LINE_SPACING      = 1.35

# ============================================================
# FIGURE-LEVEL CHROME POSITIONS (inches)
# ============================================================
# Bottom-anchored "Figure N" identifier + caption, measured up from figure bottom.
DEFAULT_FIGURE_NUMBER_Y_INCHES   = 1.05
DEFAULT_FIGURE_CAPTION_Y_INCHES  = 0.75
# Buffer added to the panel-driven bottom reservation so chrome isn't clipped.
DEFAULT_BOTTOM_PAD_BUFFER_INCHES = 0.05


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


# ============================================================
# RENDER PROFILES (named scale/path bundles)
# ============================================================
# A render profile is a NAMED bundle of style-module overrides describing a
# whole RENDER PATH: production print PNG, the live notebook (ipynb), or an
# exported GIF. It generalizes nb_compact_style() — instead of each figure
# hand-stacking context managers per path, a figure enters ONE profile and the
# entire library shares that path's scale/tone:
#
#     with render_profile("notebook"):
#         fig = build(); fig.render()
#
# Profiles carry only CROSS-CUTTING, path-level concerns (font + layout scale,
# and — as the system grows — dpi / tick direction / border model). Genuinely
# figure-specific tone (a figure's own font tier, a one-off border treatment)
# stays a figure-local override layered ON TOP, per the override ladder at the
# top of this module:  global defaults → profile → figure-local → per-object.
#
# THE PROFILES
#   "print"     base scale. Overrides are empty because the module-level
#               DEFAULT_* values ARE the print profile (the print render() path
#               builds at these defaults, never inside a profile context).
#   "notebook"  the proven compact bundle (NB_COMPACT_OVERRIDES). nb_compact_style()
#               is now a thin alias for render_profile("notebook"), so existing
#               figure show()/save_gif() call sites are byte-identical.
#   "gif"       currently the notebook scale (exported animations render at
#               notebook chrome:cell ratio). Kept as its own name so it can
#               diverge from "notebook" — e.g. a dedicated gif dpi / border
#               model — without disturbing the live-notebook look.
#
# NOTE on dpi: notebook dpi legitimately varies per figure (72 vs 80 — a
# deliberate per-figure choice), so it is NOT folded into the notebook bundle.
# A path with a single canonical dpi (print=150) reads it from DEFAULT_DPI.
# Folding a canonical gif dpi into the "gif" profile is a migration step taken
# once figures select profiles instead of hand-passing dpi.

PRINT_PROFILE_OVERRIDES = {}
NOTEBOOK_PROFILE_OVERRIDES = NB_COMPACT_OVERRIDES
GIF_PROFILE_OVERRIDES = NB_COMPACT_OVERRIDES

PROFILES = {
    "print": PRINT_PROFILE_OVERRIDES,
    "notebook": NOTEBOOK_PROFILE_OVERRIDES,
    "gif": GIF_PROFILE_OVERRIDES,
}


@_contextmanager
def render_profile(name):
    """Temporarily apply a named render profile's overrides to this module.

    `name` is one of PROFILES ("print" | "notebook" | "gif"). Same lazy-lookup
    mechanism as the legacy nb_compact_style: plottables and panels resolve
    `dsplot.style.*` at draw() time, so every figure built + rendered inside the
    context picks up the profile's scale. Originals restore on exit, so
    production paths outside any profile keep the print-scale defaults.
    """
    if name not in PROFILES:
        raise ValueError(
            f"unknown render profile {name!r}; expected one of {sorted(PROFILES)}"
        )
    _module = _sys.modules[__name__]
    overrides = PROFILES[name]
    _orig = {k: getattr(_module, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(_module, k, v)
        yield
    finally:
        for k, v in _orig.items():
            setattr(_module, k, v)


@_contextmanager
def nb_compact_style():
    """Back-compat alias for ``render_profile("notebook")``.

    Retained because every figure module's show()/save_gif() calls it directly.
    The notebook profile reuses NB_COMPACT_OVERRIDES, so this is byte-identical
    to the original implementation — just routed through the unified profile
    system so there is one place that defines "notebook scale".
    """
    with render_profile("notebook"):
        yield


# =============================================================================
# HORIZONTAL BAR CHARTS (additive — timing report figures)
# =============================================================================
# Bars are a new plottable vocabulary (Barh + BarPanel) layered on top of the
# existing panel/plottable system. These constants tune them for the shared 28"
# canvas, in the same spirit as the vector/heatmap linework above. Nothing else
# in the library reads them, so they cannot affect existing figures.
DEFAULT_BAR_HEIGHT = 0.62             # bar thickness in data (row) units
DEFAULT_BAR_EDGE_LINEWIDTH = 3.0      # separator stroke between stacked segments
DEFAULT_BAR_VALUE_FONT_SIZE = 32      # value annotation at a bar's end
DEFAULT_BAR_INBAR_FONT_SIZE = 30      # label drawn inside a (stacked) bar segment
