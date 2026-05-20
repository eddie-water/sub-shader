"""CompositePanel — one-level-nested Panel that hosts a sub-grid of child Panels.

Allows a Figure.compose row to embed a small N×M grid of child panels inside a
single outer cell. Strictly one level of nesting: a CompositePanel cannot
contain another CompositePanel (raises ValueError at construction). Row widths
inside the composite must agree, same rule as Figure.compose.

The composite's outer Axes is hidden at render time; a nested gridspec is
spawned from `ax.get_subplotspec().subgridspec(...)` and child Axes are added
to the parent matplotlib Figure. Children get the same dark facecolor /
spine styling that Figure.render() applies to top-level cells.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .. import style
from .base import Panel


class CompositePanel(Panel):
    """Panel that contains a one-level grid of child Panels.

    `rows` is a list of rows, each row a list of child Panels. Every row must
    sum to the same total `units[0]` width. Children are rendered into a
    nested gridspec carved from this Panel's outer cell.
    """

    def __init__(
        self,
        *,
        rows: List[List[Panel]],
        units: Tuple[int, int],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        share_x: bool = False,
        hspace: Optional[float] = None,
        wspace: Optional[float] = None,
    ) -> None:
        super().__init__(units=units)
        if not rows:
            raise ValueError("CompositePanel requires at least one row")

        for row in rows:
            for p in row:
                if isinstance(p, CompositePanel):
                    raise ValueError(
                        "CompositePanel cannot contain another CompositePanel — "
                        "one level of nesting only"
                    )

        widths = [sum(p.units[0] for p in row) for row in rows]
        if any(w != widths[0] for w in widths):
            raise ValueError(f"CompositePanel row widths mismatch: {widths}")

        self.rows = rows
        self.title = title
        self.subtitle = subtitle
        self.share_x = share_x
        self.hspace = hspace
        self.wspace = wspace

        if share_x:
            # Vertically stacked panels share a time/x axis — strip x decoration
            # from every row except the last so only the bottom row carries
            # xtick labels + xlabel. Done at construction so the children render
            # without ever drawing the suppressed decoration.
            last_row_idx = len(rows) - 1
            for r_idx, row in enumerate(rows):
                if r_idx == last_row_idx:
                    continue
                for p in row:
                    if hasattr(p, "x_label"):
                        p.x_label = None
                    if hasattr(p, "xticks"):
                        p.xticks = None

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("CompositePanel.render() called before attach()")

        # Local import avoids module-level cycle through panels/__init__.py.
        from .static_panel_3d import StaticPanel3D

        # Keep outer ax around so its set_title carries the composite title;
        # strip its body chrome so it doesn't draw a competing background.
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.patch.set_alpha(0.0)

        self._render_chrome_titles()

        n_rows = len(self.rows)
        n_cols = sum(p.units[0] for p in self.rows[0])

        # Convert the figure-wide inch-based gutter into matplotlib gridspec
        # fractions for THIS composite's inner cells: gap = gutter_inches /
        # inner_cell_inches, so absolute gap matches the outer grid regardless
        # of how the composite was sized.
        fig_w_in, fig_h_in = self.ax.figure.get_size_inches()
        bbox = self.ax.get_position()
        inner_cell_w = (bbox.width * fig_w_in) / n_cols
        inner_cell_h = (bbox.height * fig_h_in) / n_rows
        gutter_in = style.DEFAULT_INNER_GUTTER_INCHES
        derived_hspace = gutter_in / inner_cell_h if inner_cell_h > 0 else style.DEFAULT_HSPACE
        derived_wspace = gutter_in / inner_cell_w if inner_cell_w > 0 else style.DEFAULT_WSPACE

        subgs = self.ax.get_subplotspec().subgridspec(
            n_rows,
            n_cols,
            hspace=self.hspace if self.hspace is not None else derived_hspace,
            wspace=self.wspace if self.wspace is not None else derived_wspace,
        )

        mpl_fig = self.ax.figure

        for r in range(n_rows):
            for p in range(len(self.rows[r])):
                child = self.rows[r][p]
                col_start = sum(self.rows[r][i].units[0] for i in range(p))
                colspan = self.rows[r][p].units[0]
                cell = subgs[r, col_start:col_start + colspan]
                projection = "3d" if isinstance(child, StaticPanel3D) else None
                child_ax = mpl_fig.add_subplot(cell, projection=projection)
                child_ax.set_facecolor(style.BG_COLOR)
                if projection is None:
                    for spine in child_ax.spines.values():
                        spine.set_edgecolor(style.SPINE_COLOR)
                        spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
                child.attach(child_ax)
                child.render()
                if self.share_x and r != n_rows - 1 and projection is None:
                    # Suppress matplotlib's auto-drawn x ticks/labels on every
                    # row except the last, even if the child Panel did not set
                    # xticks itself — share_x means the bottom row owns x.
                    child_ax.tick_params(
                        axis="x", which="both",
                        bottom=False, labelbottom=False,
                    )
                    child_ax.set_xlabel("")
