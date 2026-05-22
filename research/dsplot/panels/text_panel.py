"""TextPanel — text-only Panel that renders a centered string in its cell.

Use for row labels, section headers, or any cell that should hold only
prose. The Axes is stripped of spines, ticks, and ticklabels; the cell
background fills with ``style.BG_COLOR``.

Multi-line text via embedded ``\\n``. When ``auto_shrink=True`` the font
is scaled down (never up, never below ``min_font_size``) so the rendered
text bbox fits inside the cell — text never overflows its panel unit.
"""
from __future__ import annotations

from typing import Optional, Tuple

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
        min_font_size: float = 8.0,
        cell_padding_frac: float = 0.08,
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

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("TextPanel.render() called before attach()")

        ax = self.ax
        ax.set_facecolor(style.BG_COLOR)
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
