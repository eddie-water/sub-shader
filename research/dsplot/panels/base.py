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
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

from .. import style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ..plottables.base import Plottable


class Panel(ABC):
    """Container that owns one mpl Axes and a list of Plottables."""

    ax: "Optional[Axes]"
    _plottables: "List[Plottable]"

    # Class-level default panel-unit footprint (width, height) in lego units.
    # Figure.compose reads each panel's `units` (instance override) or this
    # default to derive figsize, gridspec width, and per-panel colspan.
    default_units: ClassVar[Tuple[int, int]] = (1, 1)

    # Base reservation (subclasses override) for in-figure controls — e.g.
    # InteractivePanel sets _base_bottom_pad = 0.18 for its prev/next
    # buttons. Subtitle bottom padding is added on top at instance level
    # via the requires_bottom_pad property.
    _base_bottom_pad: float = 0.0

    @property
    def requires_bottom_pad(self) -> float:
        """Fraction of figure height to reserve below this panel.

        Combines the subclass control-bar reservation with an additional
        per-instance subtitle reservation (when self.subtitle is set), so a
        below-plot subtitle isn't clipped by Figure.savefig's tight bbox.
        Figure.render() takes max() across panels.
        """
        pad = self._base_bottom_pad
        if getattr(self, "subtitle", None) is not None:
            pad += style.DEFAULT_SUBTITLE_BOTTOM_PAD
        return pad

    def __init__(self, *, units: Optional[Tuple[int, int]] = None) -> None:
        self.units: Tuple[int, int] = units if units is not None else self.default_units
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

    def _render_chrome_titles(self) -> None:
        """Render this panel's title (above the plot) and subtitle (below).

        Library-wide convention enforced here so every Panel subclass picks
        up the same layout: title sits just above the axes box at
        ``style.DEFAULT_TITLE_Y``, subtitle sits just below the axes box at
        ``style.DEFAULT_SUBTITLE_Y``. Subclasses set ``self.title`` and
        ``self.subtitle`` and call this from their background-setup path.

        Axes3D requires ``text2D`` instead of ``text`` for axes-relative
        placement; the helper detects via ``hasattr(ax, "get_zlim")``.
        """
        title = getattr(self, "title", None)
        subtitle = getattr(self, "subtitle", None)
        ax = self.ax
        if ax is None:
            return

        if title is not None:
            # Title V-centered between the panel spine and the cell border
            # above. The chrome zone above the spine equals one PAD (= half
            # the inter-cell gutter), so the title sits PAD/2 above the
            # spine — exactly midway between the plot and the cell border.
            bbox = ax.get_position()
            fig_h_in = ax.figure.get_size_inches()[1]
            axes_h_in = max(bbox.height * fig_h_in, 0.1)
            chrome_above_in = style.DEFAULT_GUTTER_INCHES / 2.0
            y_axes = 1.0 + (chrome_above_in / 2.0) / axes_h_in
            text_fn = ax.text2D if hasattr(ax, "get_zlim") else ax.text
            text_fn(
                0.5, y_axes, title,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=style.DEFAULT_TITLE_FONT_SIZE,
                fontweight="bold",
                color=style.TICK_LABEL_COLOR,
            )

        if subtitle is not None:
            text_fn = ax.text2D if hasattr(ax, "get_zlim") else ax.text
            text_fn(
                0.5, style.DEFAULT_SUBTITLE_Y, subtitle,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=style.DEFAULT_SUBTITLE_FONT_SIZE,
                color=style.TICK_LABEL_COLOR,
                style="italic",
                fontweight="bold",
            )
