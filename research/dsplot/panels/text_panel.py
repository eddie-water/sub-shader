"""TextPanel — text-only Panel that renders a centered string in its cell.

Use for row labels, section headers, or any cell that should hold only
prose. The Axes is stripped of spines, ticks, and ticklabels; the cell
background fills with ``style.BG_COLOR``.

Multi-line text via embedded ``\\n``. When ``auto_shrink=True`` the font
is scaled down (never up, never below ``min_font_size``) so the rendered
text bbox fits inside the cell — text never overflows its panel unit.

Justify mode: pass ``justify=True`` to distribute words evenly across each
line (Microsoft Word "Justify" alignment). The justification target is the
axes (= plot-area) width minus ``cell_padding_frac`` on each side, so the
text occupies the same rectangle a plot would. Each ``\\n``-separated
paragraph is FIRST word-wrapped to fit that width at the chosen font size,
THEN every wrapped line is spread across the full width — so a long
paragraph reads as a justified block of multiple lines rather than a single
overflowing row. Auto-shrink jointly searches (size, wrapped-line-count)
for the largest font where wrapped width fits AND total stacked height
fits.
"""
from __future__ import annotations

import textwrap
from typing import List, Optional, Tuple

from .. import style
from .base import Panel


class TextPanel(Panel):
    """Panel that renders a single centered text string with no axes chrome."""

    is_text_only = True

    def __init__(
        self,
        text: str,
        *,
        units: Optional[Tuple[int, int]] = None,
        font_size: Optional[float] = None,
        color: Optional[str] = None,
        fontweight: str = "bold",
        rotation: float = 0.0,
        ha: str = "center",
        va: str = "center",
        auto_shrink: bool = True,
        min_font_size: float = 14.0,
        cell_padding_frac: float = 0.08,
        justify: bool = False,
        line_spacing: float = 1.35,
        show_ghost_border: bool = False,
        top_anchor: bool = False,
        facecolor: Optional[str] = None,
        content_margin_frac: Optional[float] = None,
    ) -> None:
        super().__init__(units=units)
        self.text = text
        self.font_size = font_size
        self.color = color
        self.fontweight = fontweight
        self.rotation = rotation
        self.ha = ha
        self.va = va
        self.auto_shrink = auto_shrink
        self.min_font_size = min_font_size
        self.cell_padding_frac = cell_padding_frac
        self.justify = justify
        self.line_spacing = line_spacing
        self.show_ghost_border = show_ghost_border
        self.top_anchor = top_anchor
        # Cell background fill. Defaults to style.BG_COLOR (opaque). Pass "none"
        # for a transparent cell so content bleeding in from an adjacent panel
        # (e.g. a neighbour's right-side tick labels rendering into this cell's
        # empty margin) isn't painted over — the figure background shows through
        # identically when it already equals BG_COLOR.
        self.facecolor = facecolor
        # When set, the justify layout ignores the (cell-spanning) content-border
        # rect and instead lays text inside the AXES box inset by this fraction on
        # every side — i.e. text starts hard in the top-left with a tiny uniform
        # margin. None keeps the legacy content-border behaviour.
        self.content_margin_frac = content_margin_frac

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("TextPanel.render() called before attach()")

        ax = self.ax
        ax.set_facecolor(self.facecolor if self.facecolor is not None else style.BG_COLOR)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        resolved_size = (
            self.font_size if self.font_size is not None
            else style.DEFAULT_TITLE_FONT_SIZE
        )
        resolved_color = (
            self.color if self.color is not None
            else style.TICK_LABEL_COLOR
        )

        if self.show_ghost_border:
            self._draw_ghost_border()

        # Use the wrap pipeline whenever justify OR top_anchor is set —
        # both require per-line layout (justify spreads words; top_anchor
        # needs to know the wrapped line count to start at the top edge).
        if self.justify or self.top_anchor:
            self._render_justified(resolved_size, resolved_color)
            return

        text_artist = ax.text(
            0.5, 0.5, self.text,
            transform=ax.transAxes,
            ha=self.ha, va=self.va,
            fontsize=resolved_size,
            color=resolved_color,
            fontweight=self.fontweight,
            rotation=self.rotation,
        )

        if self.auto_shrink:
            self._shrink_to_fit(text_artist, resolved_size)

    def _shrink_to_fit(self, text_artist, initial_font_size: float) -> None:
        """Reduce ``text_artist`` font size until its bbox fits the cell.

        Uses an Agg renderer for measurement (cheap, doesn't require a
        canvas draw). Scales by the tighter of the width/height ratios,
        with ``cell_padding_frac`` reserving a uniform inset on all sides.
        Never grows the font, and never drops below ``min_font_size``.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        ax = self.ax
        fig = ax.figure
        # FigureCanvasAgg(fig) reassigns fig.canvas; that's fine here because
        # the only canvas-dependent call afterward is fig.savefig(), which
        # works with whatever canvas is current.
        renderer = FigureCanvasAgg(fig).get_renderer()
        try:
            text_bbox = text_artist.get_window_extent(renderer)
            axes_bbox = ax.get_window_extent(renderer)
        except Exception:
            return  # measurement failure: leave font alone

        usable_w = axes_bbox.width * (1.0 - 2.0 * self.cell_padding_frac)
        usable_h = axes_bbox.height * (1.0 - 2.0 * self.cell_padding_frac)
        if text_bbox.width <= 0 or text_bbox.height <= 0:
            return

        scale_w = usable_w / text_bbox.width
        scale_h = usable_h / text_bbox.height
        scale = min(1.0, scale_w, scale_h)
        if scale >= 1.0:
            return

        new_size = max(self.min_font_size, initial_font_size * scale)
        text_artist.set_fontsize(new_size)

    def _render_justified(self, font_size: float, color: str) -> None:
        """Render text wrapped + justified across the axes width.

        Each ``\\n``-separated paragraph is word-wrapped to fit the content
        width at the chosen font size, then every wrapped line is spread so
        its words span the full content width. The LAST line of each
        paragraph is left-aligned (Word/LaTeX "Justify" convention), so a
        short final fragment doesn't fly across the cell with giant gaps.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        ax = self.ax
        fig = ax.figure
        renderer = FigureCanvasAgg(fig).get_renderer()

        try:
            axes_bbox = ax.get_window_extent(renderer)
        except Exception:
            return
        axes_w_px = axes_bbox.width
        axes_h_px = axes_bbox.height
        # Justify rect = ghost border rect (so text fills the visible content
        # frame), shrunk further by cell_padding_frac inside the ghost so the
        # text doesn't kiss the border. Horizontal extent is wider than the
        # axes when the ghost extends past the axes edges.
        if self.content_margin_frac is not None:
            gb_left, gb_bottom, gb_right, gb_top = self._content_margin_rect_axes()
            pad_frac = 0.0
        else:
            gb_left, gb_bottom, gb_right, gb_top = self._ghost_border_rect_axes()
            pad_frac = self.cell_padding_frac
        gb_w_frac = gb_right - gb_left
        gb_h_frac = gb_top - gb_bottom
        pad_w_px = axes_w_px * gb_w_frac * pad_frac
        pad_h_px = axes_h_px * gb_h_frac * pad_frac
        content_w_px = max(axes_w_px * gb_w_frac - 2.0 * pad_w_px, 1.0)
        content_h_px = max(axes_h_px * gb_h_frac - 2.0 * pad_h_px, 1.0)
        # Pixel origin of the content rect (left edge, bottom edge).
        gb_left_px = axes_bbox.x0 + axes_w_px * gb_left
        gb_bottom_px = axes_bbox.y0 + axes_h_px * gb_bottom
        gb_h_px = axes_h_px * gb_h_frac

        paragraphs = self.text.split("\n")
        size, paragraph_lines = self._fit_and_wrap(
            paragraphs, font_size, content_w_px, content_h_px, renderer
        )

        # Flatten with per-line "is last in paragraph?" flag.
        flat: List[Tuple[List[str], bool]] = []
        for para_lines in paragraph_lines:
            if not para_lines:
                flat.append(([], True))
                continue
            for idx, line in enumerate(para_lines):
                is_last = (idx == len(para_lines) - 1)
                flat.append((line.split(), is_last))

        dpi = fig.get_dpi()
        line_h_px = size * self.line_spacing * dpi / 72.0
        n_lines = len(flat)
        total_text_h_px = line_h_px * n_lines
        # Vertical anchor: top of the content rect when top_anchor=True
        # (text starts at the upper edge), otherwise centered in the rect.
        if self.top_anchor:
            top_px = gb_bottom_px + gb_h_px - pad_h_px
        else:
            top_px = gb_bottom_px + gb_h_px / 2.0 + total_text_h_px / 2.0
        left_px = gb_left_px + pad_w_px

        for i, (words, is_last) in enumerate(flat):
            y_px = top_px - (i + 0.5) * line_h_px
            y_axes = (y_px - axes_bbox.y0) / axes_h_px
            if not words:
                continue

            word_widths_px = [
                self._measure_text_px(w, size, renderer) for w in words
            ]
            total_word_px = sum(word_widths_px)
            n_gaps = len(words) - 1

            if n_gaps == 0 or is_last or not self.justify:
                # Left-aligned with a natural space gap. Fires for:
                # single-word lines (no gaps to spread), the LAST line of
                # each paragraph (Word "Justify" convention), and any line
                # when justify=False (pure left-aligned mode).
                space_px = self._measure_text_px(" ", size, renderer)
                gap_px = space_px
            else:
                gap_px = (content_w_px - total_word_px) / n_gaps

            cursor_px = left_px
            for j, word in enumerate(words):
                x_axes = (cursor_px - axes_bbox.x0) / axes_w_px
                ax.text(
                    x_axes, y_axes, word,
                    transform=ax.transAxes,
                    ha="left", va="center",
                    fontsize=size, color=color, fontweight=self.fontweight,
                )
                cursor_px += word_widths_px[j] + gap_px

    def _fit_and_wrap(
        self,
        paragraphs: List[str],
        initial_size: float,
        content_w_px: float,
        content_h_px: float,
        renderer,
    ) -> Tuple[float, List[List[str]]]:
        """Search font sizes from ``initial_size`` down to ``min_font_size``.

        At each candidate size: estimate average char width, word-wrap each
        paragraph to that char-budget, verify (a) every wrapped line's true
        rendered width fits the content width, and (b) the total stacked
        height fits. Returns the largest passing size + its wrapped lines,
        or the minimum size's wrap as fallback.
        """
        fig = self.ax.figure
        dpi = fig.get_dpi()

        size = initial_size
        last_wrap: List[List[str]] = []
        while size >= self.min_font_size:
            avg_char_w_px = max(
                self._measure_text_px("abcdefghijklmnopqrstuvwxyz", size, renderer) / 26.0,
                1.0,
            )
            chars_per_line = max(int(content_w_px / avg_char_w_px), 1)
            wrap: List[List[str]] = []
            n_total = 0
            for para in paragraphs:
                if not para.strip():
                    wrap.append([])
                    n_total += 1
                else:
                    pieces = textwrap.wrap(para, width=chars_per_line, break_long_words=False)
                    if not pieces:
                        pieces = [para]
                    wrap.append(pieces)
                    n_total += len(pieces)

            last_wrap = wrap
            # Height check.
            line_h_px = size * self.line_spacing * dpi / 72.0
            if line_h_px * n_total > content_h_px:
                size -= 1.0
                continue
            # Width check on each wrapped line (textwrap is char-budget so
            # actual proportional-font width may overshoot — verify).
            overflow = False
            for para_lines in wrap:
                for line in para_lines:
                    if self._measure_text_px(line, size, renderer) > content_w_px:
                        overflow = True
                        break
                if overflow:
                    break
            if overflow:
                size -= 1.0
                continue
            return size, wrap

        return self.min_font_size, last_wrap

    def _content_margin_rect_axes(self) -> Tuple[float, float, float, float]:
        """Text region (left, bottom, right, top) in axes-frac for the
        content_margin layout.

        Anchored to the CELL border, not the inset axes box: the axes sits
        half-a-gutter inside the visible cell border on every side, so the
        region is extended OUT to the cell border (by the measured half-gutter
        per side) then inset by ``content_margin_frac`` — making the value a
        literal "tiny margin from the cell's top-left corner" regardless of
        gutter size. Shared by the justify renderer and the ghost-border draw
        so the outlined box matches exactly where the text lays out.
        """
        half_col_frac, half_row_frac = self._half_gutter_axes_fracs()
        m = self.content_margin_frac
        left_border, right_border = self._horizontal_cell_borders_axes_frac(half_col_frac)
        return (
            left_border + m,
            -half_row_frac + m,
            right_border - m,
            1.0 + half_row_frac - m,
        )

    def _horizontal_cell_borders_axes_frac(self, half_col_frac: float) -> Tuple[float, float]:
        """Left/right CELL borders in this axes' fraction units.

        A side that has a band-sharing sibling is an interior gutter (border at
        half-a-gutter past the axes). A side with no sibling is a perimeter edge
        — its cell border is the figure edge (fig-fraction 0 or 1), which can be
        a full margin past the axes, much further than a half-gutter. Using the
        true border on each side keeps the content margin even even for an edge
        cell (e.g. the rightmost caption column, whose right border is the figure
        edge, not a half-gutter).
        """
        ax = self.ax
        my = ax.get_position()
        has_left = has_right = False
        for other in ax.figure.axes:
            if other is ax:
                continue
            ob = other.get_position()
            if not (ob.y0 < my.y1 and ob.y1 > my.y0):
                continue
            if ob.x1 <= my.x0 + 1e-6:
                has_left = True
            elif ob.x0 >= my.x1 - 1e-6:
                has_right = True
        w = max(my.width, 1e-6)
        left = -half_col_frac if has_left else (0.0 - my.x0) / w
        right = (1.0 + half_col_frac) if has_right else (1.0 - my.x0) / w
        return left, right

    def _half_gutter_axes_fracs(self) -> Tuple[float, float]:
        """Half the inter-cell gutter (col, row) in THIS axes' fraction units.

        Measures the actual spine-to-spine gap to the nearest sibling axes on
        the left and above (mirrors ``Figure._measure_half_gutter_fracs`` but
        from the panel's own vantage), then divides by two and normalizes to
        this axes' width/height so it composes with axes-fraction layout math.
        Falls back to the style gutter constants when no sibling is found
        (single-cell figure).
        """
        ax = self.ax
        fig = ax.figure
        fig_w_in, fig_h_in = fig.get_size_inches()
        my = ax.get_position()
        # Smallest spine-to-spine gap to a band-sharing sibling on EITHER side
        # (left/right for columns, above/below for rows). Taking the min picks
        # the true inter-cell gutter even for an edge cell whose only neighbor
        # in one direction is a far-away band (e.g. the top text cell's gap up
        # to the suptitle band is much larger than the body-row gutter below it).
        col_gap = None
        row_gap = None
        for other in fig.axes:
            if other is ax:
                continue
            ob = other.get_position()
            vertically_overlaps = ob.y0 < my.y1 and ob.y1 > my.y0
            horizontally_overlaps = ob.x0 < my.x1 and ob.x1 > my.x0
            if vertically_overlaps:
                if ob.x1 <= my.x0 + 1e-6:
                    gap = my.x0 - ob.x1
                elif ob.x0 >= my.x1 - 1e-6:
                    gap = ob.x0 - my.x1
                else:
                    gap = None
                if gap is not None and gap > 1e-6 and (col_gap is None or gap < col_gap):
                    col_gap = gap
            if horizontally_overlaps:
                if ob.y0 >= my.y1 - 1e-6:
                    gap = ob.y0 - my.y1
                elif ob.y1 <= my.y0 + 1e-6:
                    gap = my.y0 - ob.y1
                else:
                    gap = None
                if gap is not None and gap > 1e-6 and (row_gap is None or gap < row_gap):
                    row_gap = gap
        if col_gap is None:
            col_gap = (style.DEFAULT_COLUMN_GUTTER_INCHES / fig_w_in)
        if row_gap is None:
            row_gap = (style.DEFAULT_GUTTER_INCHES / fig_h_in)
        # col_gap/row_gap are figure fractions; convert half-gutter to this
        # axes' own fraction units (axes width/height in figure fractions).
        half_col_axes = (col_gap / 2.0) / max(my.width, 1e-6)
        half_row_axes = (row_gap / 2.0) / max(my.height, 1e-6)
        return half_col_axes, half_row_axes

    def _ghost_border_rect_axes(self) -> Tuple[float, float, float, float]:
        """The content-border rect (see ``Panel._content_border_rect_axes``).

        Kept as a thin alias so existing TextPanel call sites and the justify
        layout keep working while the geometry lives in the shared base helper
        — the "big box just inside the cell" is now common to all text panels.
        """
        return self._content_border_rect_axes()

    def _draw_ghost_border(self) -> None:
        """Outline the panel's text region.

        When ``content_margin_frac`` is set, draw the actual content-margin rect
        (so the outline matches exactly where the text lays out); otherwise fall
        back to the shared content-border rectangle.
        """
        if self.content_margin_frac is not None:
            import matplotlib.patches as mpatches

            left, bottom, right, top = self._content_margin_rect_axes()
            rect = mpatches.Rectangle(
                (left, bottom), right - left, top - bottom,
                transform=self.ax.transAxes, fill=False,
                edgecolor=style.SPINE_COLOR, linewidth=style.DEFAULT_SPINE_LINEWIDTH,
                clip_on=False, zorder=1,
            )
            self.ax.add_patch(rect)
            return
        self._draw_content_border()

    def _measure_text_px(self, text: str, size: float, renderer) -> float:
        ax = self.ax
        # Off-axes probe text — y=-1 keeps it out of the visible draw area
        # without affecting axes limits (xlim/ylim already set to [0,1]).
        probe = ax.text(
            0.0, -1.0, text,
            transform=ax.transAxes,
            fontsize=size, fontweight=self.fontweight,
        )
        try:
            bbox = probe.get_window_extent(renderer)
            width = bbox.width
        finally:
            probe.remove()
        return width
