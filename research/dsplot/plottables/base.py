"""Plottable abstract base — the contract every concrete Plottable implements.

Concrete subclasses store style knobs (color, linewidth, alpha, linestyle,
label, zorder) as plain attributes. When a knob is None at construction, the
subclass's draw() resolves it against dsplot.style.* (LAZY lookup per D-05) —
this is what makes runtime style reassignment between construction and draw
observable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from matplotlib.axes import Axes


class Plottable(ABC):
    """A unit drawn onto a matplotlib Axes.

    Style knobs are settable at construction (kwargs) and mutable after
    (attribute assignment). draw(ax) is called by the owning Panel.
    None-valued knobs resolve lazily against dsplot.style.* at draw() time.
    """

    color: str | None
    linewidth: float | None
    alpha: float
    linestyle: str
    label: str | None
    zorder: int

    def __init__(self,
                 *,
                 color: str | None = None,
                 linewidth: float | None = None,
                 alpha: float = 1.0,
                 linestyle: str = "-",
                 label: str | None = None,
                 zorder: int = 2) -> None:
        self.color = color
        self.linewidth = linewidth
        self.alpha = alpha
        self.linestyle = linestyle
        self.label = label
        self.zorder = zorder

    @abstractmethod
    def draw(self, ax: Axes) -> None:
        """Render this plottable onto ax. Pure side-effect; returns None."""
        ...
