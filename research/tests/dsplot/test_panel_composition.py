"""Panel composition contract tests — Plan 09-02 Task 1.

Verifies:
  - Panel is an abstract base class
  - StaticPanel supports fluent .add(plottable) chaining
  - StaticPanel iterates plottables in insertion order and calls .draw(ax) once each
  - axes_setup.setup_vector_axes builds square axes with optional arrow-style /
    line-style axes and optional x/y labels
  - Layout/typography defaults resolve LAZILY against dsplot.style.DEFAULT_*
    per D-05 (reassign-and-observe)

The Plottable contract is duck-typed (anything with a .draw(ax) method qualifies),
so these tests use a local _FakePlottable shim instead of coupling to 09-01's
Vector / VectorComponents — that integration is exercised at the orchestrator's
post-merge verification step.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


pytest.importorskip("dsplot", reason="dsplot package not yet available (depends on 09-01)")


class _FakePlottable:
    """Minimal Plottable stand-in for Panel-contract tests."""

    def __init__(self, marker: str = "fp"):
        self.marker = marker
        self.draw_count = 0
        self.drawn_on = None

    def draw(self, ax) -> None:
        self.draw_count += 1
        self.drawn_on = ax
        ax.plot([0, 1], [0, 1])


def _fresh_axes():
    fig, ax = plt.subplots()
    return fig, ax


def test_panel_is_abstract():
    from dsplot.panels.base import Panel
    with pytest.raises(TypeError):
        Panel()


def test_static_panel_add_returns_self_for_chaining():
    from dsplot.panels import StaticPanel
    panel = StaticPanel()
    fp = _FakePlottable()
    assert panel.add(fp) is panel


def test_static_panel_stores_plottables_in_insertion_order():
    from dsplot.panels import StaticPanel
    panel = StaticPanel()
    a, b = _FakePlottable("a"), _FakePlottable("b")
    panel.add(a).add(b)
    assert panel._plottables == [a, b]


def test_static_panel_render_calls_each_plottable_draw_once():
    from dsplot.panels import StaticPanel
    panel = StaticPanel()
    a, b = _FakePlottable("a"), _FakePlottable("b")
    panel.add(a).add(b)
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert a.draw_count == 1
        assert b.draw_count == 1
        assert a.drawn_on is ax
        assert b.drawn_on is ax
    finally:
        plt.close(fig)


def test_static_panel_title_and_subtitle_use_style_defaults():
    import dsplot.style as style
    from dsplot.panels import StaticPanel
    panel = StaticPanel(title="X", subtitle="Y")
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert ax.get_title() == "X"
        subtitle_texts = [t for t in ax.texts if t.get_text() == "Y"]
        assert subtitle_texts, "subtitle text not found on axes"
        subtitle = subtitle_texts[0]
        assert subtitle.get_fontsize() == style.DEFAULT_SUBTITLE_FONT_SIZE
        xy = subtitle.get_position()
        assert abs(xy[1] - 1.02) < 1e-6
    finally:
        plt.close(fig)


def test_static_panel_lim_sets_square_axes():
    from dsplot.panels import StaticPanel
    panel = StaticPanel(lim=1.5)
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == pytest.approx((-1.5, 1.5))
        assert ylim == pytest.approx((-1.5, 1.5))
    finally:
        plt.close(fig)


def test_setup_vector_axes_arrow_style_with_labels():
    from dsplot.axes_setup import setup_vector_axes
    fig, ax = _fresh_axes()
    try:
        before_annotations = len(ax.texts)
        setup_vector_axes(ax, axis_style="arrow", axis_labels=True)
        label_texts = [t.get_text() for t in ax.texts]
        assert "x" in label_texts
        assert "y" in label_texts
        arrow_patches = [
            child for child in ax.get_children()
            if child.__class__.__name__ == "FancyArrowPatch"
        ]
        annotations = [
            child for child in ax.get_children()
            if child.__class__.__name__ == "Annotation"
        ]
        assert len(arrow_patches) + len(annotations) >= 2
    finally:
        plt.close(fig)


def test_setup_vector_axes_line_style_uses_axhline_and_axvline():
    from dsplot.axes_setup import setup_vector_axes
    fig, ax = _fresh_axes()
    try:
        setup_vector_axes(ax, axis_style="line")
        assert len(ax.lines) >= 2, "expected axhline + axvline"
    finally:
        plt.close(fig)


def test_setup_vector_axes_resolves_lim_lazily_against_style():
    """D-05 lazy-lookup pin: setup_vector_axes(ax) with no lim arg reads
    dsplot.style.DEFAULT_VECTOR_LIM at call time."""
    import dsplot.style as style
    from dsplot.axes_setup import setup_vector_axes
    original = style.DEFAULT_VECTOR_LIM
    fig, ax = _fresh_axes()
    try:
        style.DEFAULT_VECTOR_LIM = 7.25
        setup_vector_axes(ax)
        assert ax.get_xlim() == pytest.approx((-7.25, 7.25))
        assert ax.get_ylim() == pytest.approx((-7.25, 7.25))
    finally:
        style.DEFAULT_VECTOR_LIM = original
        plt.close(fig)
