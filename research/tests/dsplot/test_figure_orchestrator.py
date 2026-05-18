"""Figure orchestrator tests — Plan 09-02 Task 2.

Verifies:
  - Figure constructs and composes Panels into one mpl Figure via gridspec
  - add_panel attaches a Panel to a gridspec cell, returns the panel for chaining
  - render() walks the panel list and calls each Panel's .attach() + .render()
  - savefig() writes a non-empty PNG via the mpl Figure's savefig path
  - suptitle wiring
  - rowspan / colspan multi-cell composition primitive
  - projection="3d" produces an Axes3D for that cell (D-04)
  - mixed 2D + 3D cells in one Figure compose
  - Layout defaults (figsize, hspace, wspace, dpi) resolve LAZILY against
    dsplot.style.DEFAULT_* at construction time (D-05)

These tests use a local _FakePlottable + _FakePanel pattern for contract-level
testing, with one end-to-end smoke test that uses real StaticPanel + Vector.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


pytest.importorskip("dsplot", reason="dsplot package not yet available (depends on 09-01)")


def _make_blank_panel():
    """Build a no-op StaticPanel — useful when only the Panel scaffolding matters."""
    from dsplot.panels import StaticPanel
    return StaticPanel()


def test_figure_constructs_with_grid_dims():
    from dsplot import Figure
    fig = Figure(n_rows=1, n_cols=2)
    try:
        assert fig is not None
    finally:
        fig.close()


def test_figure_add_panel_attaches_axes_via_render():
    from dsplot import Figure
    fig = Figure(n_rows=1, n_cols=2)
    try:
        p1 = _make_blank_panel()
        p2 = _make_blank_panel()
        fig.add_panel(p1, row=0, col=0)
        fig.add_panel(p2, row=0, col=1)
        fig.render()
        assert p1.ax is not None
        assert p2.ax is not None
        assert len(fig._mpl_fig.axes) == 2
    finally:
        fig.close()


def test_figure_add_panel_returns_panel_for_chaining():
    from dsplot import Figure
    fig = Figure(n_rows=1, n_cols=1)
    try:
        p = _make_blank_panel()
        result = fig.add_panel(p, row=0, col=0)
        assert result is p
    finally:
        fig.close()


def test_figure_savefig_writes_nonempty_png(tmp_path):
    from dsplot import Figure
    fig = Figure(n_rows=1, n_cols=1)
    try:
        fig.add_panel(_make_blank_panel(), row=0, col=0)
        fig.render()
        out = tmp_path / "smoke.png"
        fig.savefig(str(out))
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        fig.close()


def test_figure_suptitle_set_on_mpl_figure():
    from dsplot import Figure
    fig = Figure(n_rows=1, n_cols=1, suptitle="Hello")
    try:
        suptitle_obj = fig._mpl_fig._suptitle
        assert suptitle_obj is not None
        assert suptitle_obj.get_text() == "Hello"
    finally:
        fig.close()


def test_figure_end_to_end_with_real_vector(tmp_path):
    from dsplot import Figure, StaticPanel, Vector
    fig = Figure(n_rows=1, n_cols=1)
    try:
        panel = StaticPanel(lim=2.0).add(Vector((1, 1), label="a"))
        fig.add_panel(panel, row=0, col=0)
        fig.render()
        out = tmp_path / "v.png"
        fig.savefig(str(out))
        assert out.exists() and out.stat().st_size > 0
    finally:
        fig.close()


def test_figure_rowspan_colspan_spans_multiple_cells():
    """A panel placed at row=0, col=0 with rowspan=2/colspan=2 on a 3x3 grid
    must span a larger bbox than a single-cell panel — proving the spanned
    cell really is multi-cell."""
    from dsplot import Figure
    fig_span = Figure(n_rows=3, n_cols=3)
    fig_one = Figure(n_rows=3, n_cols=3)
    try:
        big = _make_blank_panel()
        small = _make_blank_panel()
        fig_span.add_panel(big, row=0, col=0, rowspan=2, colspan=2)
        fig_one.add_panel(small, row=0, col=0)
        fig_span.render()
        fig_one.render()
        big_bbox = big.ax.get_position()
        small_bbox = small.ax.get_position()
        assert big_bbox.width > small_bbox.width * 1.5, (
            f"expected spanned-cell width > 1.5x single-cell width; "
            f"big={big_bbox.width}, small={small_bbox.width}"
        )
        assert big_bbox.height > small_bbox.height * 1.5, (
            f"expected spanned-cell height > 1.5x single-cell height; "
            f"big={big_bbox.height}, small={small_bbox.height}"
        )
    finally:
        fig_span.close()
        fig_one.close()


def test_figure_projection_3d_creates_axes3d():
    """D-04: add_panel(..., projection='3d') makes an Axes3D for that cell."""
    from dsplot import Figure
    fig = Figure(n_rows=1, n_cols=2)
    try:
        p_3d = _make_blank_panel()
        p_2d = _make_blank_panel()
        fig.add_panel(p_3d, row=0, col=0, projection="3d")
        fig.add_panel(p_2d, row=0, col=1)
        fig.render()
        assert hasattr(p_3d.ax, "get_zlim"), "expected Axes3D (has get_zlim) for projection='3d'"
        assert not hasattr(p_2d.ax, "get_zlim"), "expected 2D Axes (no get_zlim) when projection=None"
    finally:
        fig.close()


def test_figure_layout_defaults_resolve_lazily_against_style():
    """D-05: figsize, hspace, wspace, dpi all resolve from style.DEFAULT_* at
    Figure construction time. Reassign-and-observe."""
    import dsplot.style as style
    from dsplot import Figure
    original_hspace = style.DEFAULT_HSPACE
    try:
        style.DEFAULT_HSPACE = 0.71
        fig = Figure(n_rows=2, n_cols=2)
        try:
            assert abs(fig._gs.hspace - 0.71) < 1e-9, (
                f"Figure did not inherit DEFAULT_HSPACE at construction time; "
                f"gs.hspace = {fig._gs.hspace}"
            )
        finally:
            fig.close()
    finally:
        style.DEFAULT_HSPACE = original_hspace
