"""Tests for the Vector (polymorphic 2D/3D per D-02) and VectorComponents
(2D only) Plottables.

D-02: ONE Vector class handles 2-tuple and 3-tuple input.
D-06: Asymmetric dispatch — 3-tuple onto 2D Axes raises TypeError; 2-tuple
      onto Axes3D silently extends to z=0.
D-05: Lazy style lookup — reassigning a None-defaulted constant between
      construct and draw uses the NEW value.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


def _fresh_2d():
    fig, ax = plt.subplots()
    return fig, ax


def _fresh_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def test_vector_accepts_2tuple():
    from dsplot import Vector
    v = Vector((1.0, 0.0))
    assert tuple(v.vec) == (1.0, 0.0)


def test_vector_accepts_3tuple():
    from dsplot import Vector
    v = Vector((1.0, 0.0, 0.0))
    assert tuple(v.vec) == (1.0, 0.0, 0.0)


def test_vector_rejects_invalid_length():
    from dsplot import Vector
    with pytest.raises(ValueError, match="2-tuple or 3-tuple"):
        Vector((1.0, 2.0, 3.0, 4.0))


def test_vector_2d_draw_adds_artists():
    from dsplot import Vector
    fig, ax = _fresh_2d()
    try:
        before = len(ax.patches) + len(ax.lines) + len(ax.texts)
        Vector((1.0, 0.0)).draw(ax)
        after = len(ax.patches) + len(ax.lines) + len(ax.texts)
        assert after > before
    finally:
        plt.close(fig)


def test_vector_2d_explicit_color():
    from dsplot import Vector
    from matplotlib.colors import to_rgba
    fig, ax = _fresh_2d()
    try:
        Vector((1.0, 0.0), color="#ff0000").draw(ax)
        patch_colors = [p.get_edgecolor() for p in ax.patches]
        line_colors = [l.get_color() for l in ax.lines]
        target = to_rgba("#ff0000")
        match = any(
            tuple(round(c, 4) for c in (color if isinstance(color, tuple) else to_rgba(color))) ==
            tuple(round(c, 4) for c in target)
            for color in patch_colors + line_colors
        )
        assert match, f"expected #ff0000 on at least one artist; patches={patch_colors} lines={line_colors}"
    finally:
        plt.close(fig)


def test_vector_2d_lazy_color_lookup():
    """D-05 lazy lookup: reassign style.PRIMARY_COLOR between construct + draw."""
    import dsplot.style as style
    from dsplot import Vector
    from matplotlib.colors import to_rgba
    original = style.PRIMARY_COLOR
    fig, ax = _fresh_2d()
    try:
        v = Vector((1.0, 0.0))
        style.PRIMARY_COLOR = "#00ff00"
        v.draw(ax)
        target = to_rgba("#00ff00")
        artists = [
            p.get_edgecolor() for p in ax.patches
        ] + [l.get_color() for l in ax.lines]
        rounded_artist_rgba = [
            tuple(round(c, 4) for c in (a if isinstance(a, tuple) else to_rgba(a)))
            for a in artists
        ]
        assert tuple(round(c, 4) for c in target) in rounded_artist_rgba, (
            f"expected lazy resolution to #00ff00; got {artists}"
        )
    finally:
        style.PRIMARY_COLOR = original
        plt.close(fig)


def test_vector_2d_label_renders_text():
    from dsplot import Vector
    fig, ax = _fresh_2d()
    try:
        Vector((1.0, 0.0), label="a").draw(ax)
        labels = [t.get_text() for t in ax.texts]
        assert "a" in labels
    finally:
        plt.close(fig)


def test_vector_2d_dashed_uses_split_artists():
    """Dashed shafts render via ax.plot + a SOLID FancyArrowPatch head stub —
    sidesteps the dashed-arrowhead ghosting workaround."""
    from dsplot import Vector
    fig, ax = _fresh_2d()
    try:
        Vector((1.0, 0.0), linestyle="--").draw(ax)
        dashed_lines = [l for l in ax.lines if l.get_linestyle() in ("--", "dashed")]
        assert dashed_lines, "expected at least one dashed Line2D for the shaft"
        solid_patches = [
            p for p in ax.patches
            if hasattr(p, "get_linestyle") and p.get_linestyle() in ("-", "solid")
        ]
        assert solid_patches, "expected at least one solid FancyArrowPatch head"
    finally:
        plt.close(fig)


def test_vector_3d_draw_adds_line():
    from dsplot import Vector
    fig, ax = _fresh_3d()
    try:
        Vector((1.0, 1.0, 1.0)).draw(ax)
        assert len(ax.lines) >= 1
    finally:
        plt.close(fig)


def test_vector_3d_show_tip_toggles_scatter():
    from dsplot import Vector
    fig, ax = _fresh_3d()
    try:
        Vector((1.0, 1.0, 1.0), show_tip=True).draw(ax)
        with_tip = len(ax.collections)
        plt.close(fig)
        fig2, ax2 = _fresh_3d()
        Vector((1.0, 1.0, 1.0), show_tip=False).draw(ax2)
        without_tip = len(ax2.collections)
        plt.close(fig2)
        assert with_tip > without_tip
    except Exception:
        plt.close(fig)
        raise


def test_vector_3d_label_renders():
    from dsplot import Vector
    fig, ax = _fresh_3d()
    try:
        Vector((1.0, 1.0, 1.0), label="a").draw(ax)
        labels = [t.get_text() for t in ax.texts]
        assert "a" in labels
    finally:
        plt.close(fig)


def test_vector_3tuple_on_2d_axes_raises():
    """D-06: no defensible 3D→2D collapse → loud TypeError."""
    from dsplot import Vector
    fig, ax = _fresh_2d()
    try:
        with pytest.raises(TypeError, match=r"3D"):
            Vector((1.0, 1.0, 1.0)).draw(ax)
    finally:
        plt.close(fig)


def test_vector_2tuple_on_3d_axes_extends_to_z_zero():
    """D-06: 2-tuple on Axes3D silently extends to (x, y, 0)."""
    from dsplot import Vector
    fig, ax = _fresh_3d()
    try:
        Vector((1.0, 0.0)).draw(ax)
        assert len(ax.lines) >= 1
        line = ax.lines[-1]
        z_data = list(line.get_data_3d()[2])
        assert z_data == [0.0, 0.0], f"expected z=[0,0]; got {z_data}"
    finally:
        plt.close(fig)


def test_vector_origin_default_dimension_matches_input():
    from dsplot import Vector
    v2 = Vector((1.0, 0.0))
    v3 = Vector((1.0, 0.0, 0.0))
    assert tuple(v2.origin) == (0.0, 0.0)
    assert tuple(v3.origin) == (0.0, 0.0, 0.0)


def test_vector_components_x_first_renders_two_arrows_and_two_droplines():
    from dsplot import VectorComponents
    fig, ax = _fresh_2d()
    try:
        VectorComponents((2.0, 3.0), first_axis="x").draw(ax)
        dashed_lines = [l for l in ax.lines if l.get_linestyle() in ("--", "dashed")]
        # 2 component-arrow shafts (dashed) + 2 droplines (dashed) = 4
        assert len(dashed_lines) >= 4, f"expected >=4 dashed lines, got {len(dashed_lines)}"
    finally:
        plt.close(fig)


def test_vector_components_y_first_renders_y_then_x():
    """y-first ordering: y-arrow from origin, then x-arrow from y-arrow's tip."""
    from dsplot import VectorComponents
    fig, ax = _fresh_2d()
    try:
        VectorComponents((2.0, 3.0), first_axis="y", show_droplines=False).draw(ax)
        dashed_lines = [l for l in ax.lines if l.get_linestyle() in ("--", "dashed")]
        assert len(dashed_lines) >= 2
        # First arrow shaft goes from origin to (0, 3.0) (y-first)
        first = dashed_lines[0]
        xs, ys = first.get_xdata(), first.get_ydata()
        assert list(xs) == [0.0, 0.0]
        assert list(ys) == [0.0, 3.0]
    finally:
        plt.close(fig)


def test_vector_components_no_droplines_when_disabled():
    from dsplot import VectorComponents
    fig, ax = _fresh_2d()
    try:
        VectorComponents((2.0, 3.0), show_droplines=False).draw(ax)
        dashed_lines = [l for l in ax.lines if l.get_linestyle() in ("--", "dashed")]
        # Only 2 dashed component-arrow shafts; no extra dropline pair
        assert len(dashed_lines) == 2, f"expected exactly 2 dashed lines, got {len(dashed_lines)}"
    finally:
        plt.close(fig)
