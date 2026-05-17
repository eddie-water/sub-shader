"""Panel abstract base — container that owns one mpl Axes and a list of Plottables.

Concrete subclasses (StaticPanel / future DynamicPanel / InteractivePanel)
differ only in WHEN they call render() — the composition contract (`add`,
`attach`, `render`) is shared.

Lifecycle:
  1. Caller constructs the Panel with layout kwargs (title, lim, etc.).
  2. Caller calls `panel.add(plottable)` zero or more times.
  3. The owning Figure calls `panel.attach(ax)` when placing the panel into
     a gridspec cell.
  4. The owning Figure calls `panel.render()` to draw all plottables onto
     the attached axes.

A caller that invokes `render()` before `attach()` hits a RuntimeError — this
is intentional. Figure.render() handles the attach + render handshake.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ..plottables.base import Plottable


class Panel(ABC):
    """Container that owns one mpl Axes and a list of Plottables."""

    ax: "Optional[Axes]"
    _plottables: "List[Plottable]"

    # Fraction of figure height that this panel needs reserved below its
    # gridspec cell for in-figure controls (mpl widgets etc.). The Figure
    # orchestrator takes the max across panels and applies
    # `subplots_adjust(bottom=...)`. Default 0 means no extra reservation.
    requires_bottom_pad: float = 0.0

    def __init__(self) -> None:
        self.ax = None
        self._plottables = []

    def add(self, plottable: "Plottable") -> "Panel":
        self._plottables.append(plottable)
        return self

    def attach(self, ax: "Axes") -> None:
        self.ax = ax

    @abstractmethod
    def render(self) -> None:
        ...
