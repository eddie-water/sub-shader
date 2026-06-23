"""RichText Plottable — a single line of text with per-segment colors.

A line is given as a list of ``(text, color)`` segments; the segments are laid
out left-to-right (via matplotlib's offsetbox HPacker, which measures and packs
them) and the whole run is centered on ``xy``. Use it when one line mixes
colors — e.g. an equation where the ``a`` terms are orange, the ``b`` terms are
purple, and the operators / result are white.

Placement mirrors Annotation: ``transform="axes"`` (default) anchors ``xy`` in
axes-fraction coords, ``transform="data"`` in data coords. None-valued segment
colors and ``fontsize`` resolve against dsplot.style at draw() time.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from matplotlib.axes import Axes

from .. import style
from .base import Plottable


_VALID_TRANSFORMS = ("data", "axes")


class RichText(Plottable):
    """One line of text whose segments each carry their own color."""

    def __init__(self,
                 segments: Sequence[Tuple[str, Optional[str]]],
                 xy: Tuple[float, float],
                 *,
                 fontsize: Optional[float] = None,
                 fontweight: str = "bold",
                 fontfamily: Optional[str] = None,
                 transform: str = "axes",
                 align: str = "baseline",
                 box_alignment: Tuple[float, float] = (0.5, 0.5),
                 zorder: int = 5) -> None:
        if transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORMS}, got {transform!r}"
            )
        super().__init__(color=None, linewidth=None, alpha=1.0,
                         linestyle="-", label=None, zorder=zorder)
        self.segments: List[Tuple[str, Optional[str]]] = list(segments)
        self.xy = tuple(xy)
        self.fontsize = fontsize
        self.fontweight = fontweight
        self.fontfamily = fontfamily
        self.transform = transform
        self.align = align
        self.box_alignment = box_alignment

    def draw(self, ax: Axes) -> None:
        from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

        fontsize = (
            self.fontsize if self.fontsize is not None
            else style.DEFAULT_LABEL_FONT_SIZE
        )
        base_props = dict(fontsize=fontsize, fontweight=self.fontweight)
        if self.fontfamily is not None:
            base_props["fontfamily"] = self.fontfamily
        areas = [
            TextArea(
                text,
                textprops=dict(
                    color=(color if color is not None else style.TICK_LABEL_COLOR),
                    **base_props,
                ),
            )
            for text, color in self.segments
        ]
        packed = HPacker(children=areas, pad=0, sep=0, align=self.align)
        xycoords = "axes fraction" if self.transform == "axes" else "data"
        box = AnnotationBbox(
            packed,
            self.xy,
            xycoords=xycoords,
            frameon=False,
            pad=0.0,
            box_alignment=self.box_alignment,
            zorder=self.zorder,
        )
        ax.add_artist(box)
