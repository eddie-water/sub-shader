"""SuptitlePanel — text-only Panel for figure-level supertitle text.

A subclass of TextPanel with distinct style defaults (SUPTITLE_FONT_SIZE /
SUPTITLE_COLOR / SUPTITLE_WEIGHT instead of TITLE_FONT_SIZE / TICK_LABEL_COLOR)
and `auto_shrink=False` by default (suptitle text is single-line and should
not be shrunk to fit a narrow cell).

Two ways to use it:

  * Sugar path — Figure.compose(suptitle="…"): the legacy
    `_mpl_fig.suptitle(...)` rendering path is preserved bit-identically.
    SuptitlePanel is not directly involved in rendering, but the
    SUPTITLE_* style constants still drive that path's visual contract by
    mirroring the existing render values.

  * Explicit path — rows=[[SuptitlePanel("…", units=(N,1))], …]: the
    SuptitlePanel participates in the panel-grid as a real first-row Panel,
    renders via TextPanel.render(), and is validated by the existing
    width-equality check in Figure.compose.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .. import style
from .text_panel import TextPanel


class SuptitlePanel(TextPanel):
    """Figure-level suptitle as a real Panel (subclass of TextPanel).

    Carries SUPTITLE_* style defaults rather than TextPanel's title defaults,
    and disables auto_shrink by default (a single-line suptitle should not
    be scaled down to fit a narrow cell).
    """

    def __init__(
        self,
        text: str,
        *,
        units: Optional[Tuple[int, int]] = None,
        font_size: Optional[float] = None,
        color: Optional[str] = None,
        fontweight: Optional[str] = None,
        auto_shrink: bool = False,
    ) -> None:
        resolved_font_size = (
            font_size if font_size is not None else style.SUPTITLE_FONT_SIZE
        )
        resolved_color = (
            color if color is not None else style.SUPTITLE_COLOR
        )
        resolved_fontweight = (
            fontweight if fontweight is not None else style.SUPTITLE_WEIGHT
        )
        super().__init__(
            text,
            units=units,
            font_size=resolved_font_size,
            color=resolved_color,
            fontweight=resolved_fontweight,
            auto_shrink=auto_shrink,
            # Optical V-centering: matplotlib's va="center" aligns on the
            # bbox midline (includes descender slack even when the string
            # has none) — a string like "Fourier vs Wavelet Decomposition"
            # ends up visibly low. "center_baseline" pivots on the line
            # baseline midline instead and reads as true V-center.
            va="center_baseline",
        )
