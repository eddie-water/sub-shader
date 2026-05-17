"""Tests for the Annotation Plottable (2D, with axes-relative transform support).

Annotation supports both data-coord placement (default) and axes-relative
placement (transform="axes") — the latter enables the "result text below
the panel" pattern used by dot_product_geometry without coupling the figure
to data limits.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


def _fresh():
    fig, ax = plt.subplots()
    return fig, ax


def test_annotation_renders_text_at_xy():
    from dsplot import Annotation
    fig, ax = _fresh()
    try:
        Annotation("hello", (0.5, 0.5)).draw(ax)
        texts = [t for t in ax.texts if t.get_text() == "hello"]
        assert len(texts) == 1
        xy = texts[0].get_position()
        assert xy == pytest.approx((0.5, 0.5))
    finally:
        plt.close(fig)


def test_annotation_with_arrow_callout_adds_arrow():
    from dsplot import Annotation
    fig, ax = _fresh()
    try:
        before_annotations = sum(
            1 for c in ax.get_children() if c.__class__.__name__ == "Annotation"
        )
        Annotation("hello", (0.5, 0.5), arrow_to=(1.0, 1.0)).draw(ax)
        after_annotations = sum(
            1 for c in ax.get_children() if c.__class__.__name__ == "Annotation"
        )
        assert after_annotations > before_annotations
        ann_texts = [
            c for c in ax.get_children()
            if c.__class__.__name__ == "Annotation" and c.get_text() == "hello"
        ]
        assert ann_texts, "expected ax.annotate-created Annotation with our text"
    finally:
        plt.close(fig)


def test_annotation_transform_axes_uses_transAxes():
    from dsplot import Annotation
    fig, ax = _fresh()
    try:
        Annotation("legend", (0.5, -0.08), transform="axes").draw(ax)
        target = [t for t in ax.texts if t.get_text() == "legend"]
        assert target
        assert target[0].get_transform() is ax.transAxes
    finally:
        plt.close(fig)


def test_annotation_invalid_transform_raises():
    from dsplot import Annotation
    fig, ax = _fresh()
    try:
        with pytest.raises(ValueError, match=r"transform"):
            Annotation("oops", (0.0, 0.0), transform="figure").draw(ax)
    finally:
        plt.close(fig)


def test_annotation_defaults_resolve_lazily_against_style():
    import dsplot.style as style
    from dsplot import Annotation
    from matplotlib.colors import to_rgba
    fig, ax = _fresh()
    original_color = style.TICK_LABEL_COLOR
    original_size = style.DEFAULT_LABEL_FONT_SIZE
    try:
        a = Annotation("x", (0.0, 0.0))
        style.TICK_LABEL_COLOR = "#0000ff"
        style.DEFAULT_LABEL_FONT_SIZE = 27
        a.draw(ax)
        target = [t for t in ax.texts if t.get_text() == "x"][0]
        target_rgba = tuple(round(c, 4) for c in to_rgba("#0000ff"))
        actual_rgba = tuple(round(c, 4) for c in to_rgba(target.get_color()))
        assert actual_rgba == target_rgba
        assert target.get_fontsize() == 27
    finally:
        style.TICK_LABEL_COLOR = original_color
        style.DEFAULT_LABEL_FONT_SIZE = original_size
        plt.close(fig)


def test_top_level_imports_include_annotation():
    from dsplot import Vector, VectorComponents, Annotation
    assert Vector is not None
    assert VectorComponents is not None
    assert Annotation is not None
