"""Mixed-type Figure composition tests — Plan 09-04 Task 3 (automated portion).

These tests pin the programmatic guarantees that mixed Static + Dynamic /
Static + Interactive Figures compose without crashing. The wall-clock
visual verification (animation actually loops; widget controls respond
to clicks) is the human-verify checkpoint — pytest can't observe a live
FuncAnimation timer or an ipywidgets event loop.

Per LOCKED non-goal #6, if the headless smoke fails, the fallback is
"same-type-only Figures + document in SUMMARY." This file documents the
WORKING headless path.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


pytest.importorskip("dsplot", reason="dsplot package not yet available")


def test_mixed_static_dynamic_figure_renders():
    """Static panel + Dynamic panel side by side on one Figure.

    Pins:
      - Figure.render() walks both panels without crashing
      - Each panel's plottables actually draw onto their own axes
      - Figure._anim (the master clock that drives every DynamicPanel) is
        held on the Figure after render()
    """
    from dsplot import Figure, StaticPanel, DynamicPanel, Vector

    fig = Figure(n_rows=1, n_cols=2, figsize=(10, 5))
    static_panel = StaticPanel(
        title="Static", lim=1.5, axis_style="arrow",
    ).add(Vector((0.8, 0.6), label="a"))
    frames = [
        [Vector((np.cos(t), np.sin(t)), label="b")]
        for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ]
    dyn_panel = DynamicPanel(
        frames=frames, interval_ms=300, lim=1.5,
        axis_style="arrow", title="Dynamic",
    )
    fig.add_panel(static_panel, row=0, col=0)
    fig.add_panel(dyn_panel, row=0, col=1)

    try:
        fig.render()
        static_labels = [t.get_text() for t in static_panel.ax.texts]
        dyn_labels = [t.get_text() for t in dyn_panel.ax.texts]
        assert "a" in static_labels, f"static panel missing 'a': {static_labels}"
        assert "b" in dyn_labels, f"dynamic panel missing 'b': {dyn_labels}"
        assert fig._anim is not None, (
            "Figure master FuncAnimation reference dropped — would freeze on frame 0"
        )
        assert dyn_panel._anim is None, (
            "DynamicPanel inside a Figure should defer to the figure clock; "
            "its own FuncAnimation must not be created."
        )
    finally:
        fig.close()


def test_mixed_static_interactive_figure_renders():
    """Static panel + Interactive panel side by side; Interactive falls back
    to static frame 0 if ipywidgets is absent. Either way, both axes receive
    their content."""
    from dsplot import Figure, StaticPanel, InteractivePanel, Vector

    fig = Figure(n_rows=1, n_cols=2, figsize=(10, 5))
    static_panel = StaticPanel(
        title="ref", lim=1.5, axis_style="arrow",
    ).add(Vector((1, 0), label="ref"))
    inter_frames = [
        [Vector((x, 0.5), label=f"x={x:.1f}")] for x in [-1.0, -0.5, 0.0, 0.5, 1.0]
    ]
    inter_panel = InteractivePanel(
        frames=inter_frames, lim=1.5, axis_style="arrow", title="step",
    )
    fig.add_panel(static_panel, row=0, col=0)
    fig.add_panel(inter_panel, row=0, col=1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.render()
        static_labels = [t.get_text() for t in static_panel.ax.texts]
        inter_labels = [t.get_text() for t in inter_panel.ax.texts]
        assert "ref" in static_labels
        assert any(label.startswith("x=") for label in inter_labels), (
            f"interactive panel missing frame-0 label: {inter_labels}"
        )
    finally:
        fig.close()


def test_mixed_figure_savefig_writes_png(tmp_path):
    """Mixed Static+Dynamic Figure can savefig — one of the consumer
    workflows (export a single still frame from a mixed figure for README)."""
    from dsplot import Figure, StaticPanel, DynamicPanel, Vector

    fig = Figure(n_rows=1, n_cols=2, figsize=(10, 5))
    fig.add_panel(
        StaticPanel(title="left").add(Vector((1, 0), label="s")),
        row=0, col=0,
    )
    fig.add_panel(
        DynamicPanel(
            frames=[[Vector((np.cos(t), np.sin(t)), label="d")] for t in (0, 1, 2)],
            interval_ms=200, lim=1.5,
        ),
        row=0, col=1,
    )

    try:
        fig.render()
        out_path = tmp_path / "mixed.png"
        result = fig.savefig(str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert isinstance(result, str)
    finally:
        fig.close()
