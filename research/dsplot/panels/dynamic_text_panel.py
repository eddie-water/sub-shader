"""DynamicTextPanel — a DynamicPanel whose canvas is clean text, not vector axes.

Reuses every bit of DynamicPanel's frame/tick/master-clock machinery (so it
animates in lockstep with sibling DynamicPanels under the figure's master
clock) but swaps the background: instead of the vector-axes crosshair / ticks /
grid / aspect-equal box that ``DynamicPanel._render_background`` installs, this
panel sets up a TextPanel-style canvas — [0, 1] axes coords, no aspect
constraint, no ticks. Frame plottables are expected to be
``Annotation(transform="axes")`` so they place in axes-fraction coords and
update each frame like any other DynamicPanel frame content.

Use for live-updating math / readout panels that must stay synced with the
plots they explain (e.g. a "dot product two ways" panel beside the geometry it
narrates).
"""
from __future__ import annotations

from .. import style
from .dynamic_panel import DynamicPanel


class DynamicTextPanel(DynamicPanel):
    """DynamicPanel that renders per-frame text on a clean (axis-free) canvas."""

    def _render_background(self) -> None:
        """Clean text canvas: bg fill, optional content border, [0,1] coords.

        Mirrors TextPanel's static setup but keeps DynamicPanel's per-frame
        animate/tick path intact — frame plottables draw on top each tick and
        are cleared via the inherited tracked-artist removal. ``show_border``
        draws the SHARED content border (the big box just inside the cell, same
        as TextPanel / figure 1) rather than the tight axes spine, so live-text
        panels get the same generous content frame as every other text panel.
        """
        ax = self.ax
        ax.set_facecolor(style.BG_COLOR)
        # The tight axes spine is always hidden — text panels frame their
        # content with the larger shared content border instead.
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        if self.show_border:
            self._draw_content_border()

        self._render_chrome_titles()

        for plottable in self.base_plottables:
            plottable.draw(ax)
