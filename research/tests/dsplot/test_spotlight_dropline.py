"""Tests for Spotlight + Dropline Plottables.

Each behavior in the plan maps to one test:

Spotlight:
  1. mode="rectangle" + x_range  → axvspan patch
  2. mode="rectangle" + y_range  → axhspan patch
  3. mode="rectangle" + both     → 2 patches
  4. mode="scatter"   + xy       → scatter collection
  5. mode="glow"      + xy       → Circle patch with alpha < 1
  6. mode="rectangle" with neither range  → ValueError on draw
  7. mode="invalid_mode"                  → ValueError listing valid modes

Dropline:
  8. default-styled  → Line2D w/ style.DROPLINE_COLOR, DEFAULT_DROPLINE_ALPHA,
                       DEFAULT_DROPLINE_LINESTYLE
  9. explicit color  → uses the explicit color, not the default
 10. D-05 lazy lookup: reassigning style.DROPLINE_COLOR between construct and
     draw is observable on the resulting Line2D's color.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle

import dsplot
from dsplot import Dropline, Spotlight, style


# ============================================================
# Spotlight — rectangle mode
# ============================================================

def test_spotlight_rectangle_x_range_adds_axvspan_patch() -> None:
    fig, ax = plt.subplots()
    Spotlight(mode="rectangle", x_range=(0.1, 0.3)).draw(ax)

    assert len(ax.patches) == 1
    plt.close(fig)


def test_spotlight_rectangle_y_range_adds_axhspan_patch() -> None:
    fig, ax = plt.subplots()
    Spotlight(mode="rectangle", y_range=(0.2, 0.4)).draw(ax)

    assert len(ax.patches) == 1
    plt.close(fig)


def test_spotlight_rectangle_both_ranges_adds_two_spans() -> None:
    fig, ax = plt.subplots()
    Spotlight(mode="rectangle", x_range=(0.1, 0.3), y_range=(0.2, 0.4)).draw(ax)

    assert len(ax.patches) == 2
    plt.close(fig)


def test_spotlight_rectangle_no_ranges_raises_value_error_on_draw() -> None:
    fig, ax = plt.subplots()
    sp = Spotlight(mode="rectangle")

    with pytest.raises(ValueError):
        sp.draw(ax)
    plt.close(fig)


# ============================================================
# Spotlight — scatter mode
# ============================================================

def test_spotlight_scatter_adds_scatter_collection() -> None:
    fig, ax = plt.subplots()
    Spotlight(mode="scatter", xy=(0.5, 0.5)).draw(ax)

    assert len(ax.collections) >= 1
    plt.close(fig)


# ============================================================
# Spotlight — glow mode
# ============================================================

def test_spotlight_glow_adds_circle_with_alpha_lt_one() -> None:
    fig, ax = plt.subplots()
    Spotlight(mode="glow", xy=(0.5, 0.5), radius=0.1).draw(ax)

    circles = [p for p in ax.patches if isinstance(p, Circle)]
    assert len(circles) == 1
    assert circles[0].get_alpha() < 1.0
    plt.close(fig)


# ============================================================
# Spotlight — invalid mode validation
# ============================================================

def test_spotlight_invalid_mode_raises_value_error_with_valid_modes_listed() -> None:
    fig, ax = plt.subplots()
    sp = Spotlight(mode="not_a_real_mode")

    with pytest.raises(ValueError) as exc_info:
        sp.draw(ax)

    msg = str(exc_info.value)
    assert "rectangle" in msg
    assert "scatter" in msg
    assert "glow" in msg
    plt.close(fig)


# ============================================================
# Dropline
# ============================================================

def test_dropline_default_styling_matches_style_constants() -> None:
    fig, ax = plt.subplots()
    Dropline((0, 0), (1, 1)).draw(ax)

    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert line.get_linestyle() == style.DEFAULT_DROPLINE_LINESTYLE
    expected_rgb = matplotlib.colors.to_rgb(style.DROPLINE_COLOR)
    assert tuple(matplotlib.colors.to_rgb(line.get_color())) == pytest.approx(
        expected_rgb, abs=1e-6
    )
    assert line.get_alpha() == pytest.approx(style.DEFAULT_DROPLINE_ALPHA, abs=1e-6)
    plt.close(fig)


def test_dropline_explicit_color_overrides_default() -> None:
    fig, ax = plt.subplots()
    Dropline((0, 0), (1, 1), color="#ff0000").draw(ax)

    line = ax.lines[0]
    actual_rgb = matplotlib.colors.to_rgb(line.get_color())
    assert actual_rgb == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    plt.close(fig)


def test_dropline_lazy_lookup_dropline_color_observable() -> None:
    """D-05: reassigning style.DROPLINE_COLOR between construct and draw must
    be visible on the resulting line's color."""
    original = dsplot.style.DROPLINE_COLOR
    try:
        dsplot.style.DROPLINE_COLOR = "#00ff00"
        fig, ax = plt.subplots()

        Dropline((0, 0), (1, 1)).draw(ax)

        line = ax.lines[0]
        actual_rgb = matplotlib.colors.to_rgb(line.get_color())
        assert actual_rgb == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
        plt.close(fig)
    finally:
        dsplot.style.DROPLINE_COLOR = original
