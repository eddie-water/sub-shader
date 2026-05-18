"""InteractivePanel tests — Plan 09-04 Task 2 (mpl-widgets pivot).

Verifies the InteractivePanel behaviors after the mpl-widgets migration:
  1. constructs with frames list
  2. empty frames → ValueError on construct
  3. attach+render draws frame 0 to ax (artist count > 0)
  4. _set_frame(1) swaps frame artists
  5. render in Agg backend completes without error and creates prev/next widgets
  6. prev/next are matplotlib.widgets.Button instances; clicking advances
  7. slider=True → matplotlib.widgets.Slider added; set_val jumps frame
  8. checkbox=("label", cb) → matplotlib.widgets.CheckButtons added; toggle fires cb
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import pytest


pytest.importorskip("dsplot", reason="dsplot package not yet available")


def _fresh_axes():
    fig, ax = plt.subplots()
    return fig, ax


def test_interactive_panel_constructs_with_frames():
    from dsplot import InteractivePanel, Vector
    panel = InteractivePanel(frames=[[Vector((1, 0))], [Vector((0, 1))]])
    assert panel is not None


def test_interactive_panel_empty_frames_raises_value_error():
    from dsplot import InteractivePanel
    with pytest.raises(ValueError):
        InteractivePanel(frames=[])


def test_interactive_panel_render_draws_frame_zero_to_axes():
    from dsplot import InteractivePanel, Vector
    panel = InteractivePanel(
        frames=[[Vector((1, 0), label="f0")], [Vector((0, 1), label="f1")]]
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        labels = [t.get_text() for t in ax.texts]
        assert "f0" in labels
        assert "f1" not in labels
        total_artists = (
            len(ax.patches) + len(ax.lines) + len(ax.texts)
            + len(ax.collections) + len(ax.images)
        )
        assert total_artists > 0
    finally:
        plt.close(fig)


def test_interactive_panel_set_frame_swaps_artists():
    from dsplot import InteractivePanel, Vector
    panel = InteractivePanel(
        frames=[[Vector((1, 0), label="f0")], [Vector((0, 1), label="f1")]]
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        panel._set_frame(1)
        labels = [t.get_text() for t in ax.texts]
        assert "f1" in labels
        assert "f0" not in labels
        panel._set_frame(0)
        labels = [t.get_text() for t in ax.texts]
        assert "f0" in labels
        assert "f1" not in labels
    finally:
        plt.close(fig)


def test_interactive_panel_creates_prev_next_buttons():
    """In Agg backend, mpl Button widgets are created and exposed on
    panel._widgets — no ipywidgets dependency, no warning."""
    from dsplot import InteractivePanel, Vector
    panel = InteractivePanel(
        frames=[[Vector((1, 0))], [Vector((0, 1))]]
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert "prev" in panel._widgets
        assert "next" in panel._widgets
        assert isinstance(panel._widgets["prev"], mwidgets.Button)
        assert isinstance(panel._widgets["next"], mwidgets.Button)
    finally:
        plt.close(fig)


def test_interactive_panel_next_button_advances_frame():
    """Invoking the connected Next button callback steps the current frame."""
    from dsplot import InteractivePanel, Vector
    panel = InteractivePanel(
        frames=[[Vector((1, 0), label="f0")],
                [Vector((0, 1), label="f1")],
                [Vector((-1, 0), label="f2")]]
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert panel._current_frame == 0
        panel._widgets["next"]._observers.process("clicked", None)
        assert panel._current_frame == 1
        panel._widgets["next"]._observers.process("clicked", None)
        assert panel._current_frame == 2
        # Past end: stays clamped
        panel._widgets["next"]._observers.process("clicked", None)
        assert panel._current_frame == 2
        panel._widgets["prev"]._observers.process("clicked", None)
        assert panel._current_frame == 1
    finally:
        plt.close(fig)


def test_interactive_panel_slider_kwarg_adds_slider_widget():
    """slider=True attaches a mpl Slider; set_val jumps to the chosen frame."""
    from dsplot import InteractivePanel, Vector
    panel = InteractivePanel(
        frames=[[Vector((1, 0), label="f0")],
                [Vector((0, 1), label="f1")],
                [Vector((-1, 0), label="f2")]],
        slider=True,
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert "slider" in panel._widgets
        assert isinstance(panel._widgets["slider"], mwidgets.Slider)
        assert panel._widgets["slider"].valmax == 2
        panel._widgets["slider"].set_val(2)
        assert panel._current_frame == 2
    finally:
        plt.close(fig)


def test_interactive_panel_checkbox_kwarg_wires_callback():
    """checkbox=(label, cb) attaches a mpl CheckButtons widget; toggle fires
    the callback with the new boolean state."""
    from dsplot import InteractivePanel, Vector
    captured = []

    def cb(value: bool) -> None:
        captured.append(value)

    panel = InteractivePanel(
        frames=[[Vector((1, 0))], [Vector((0, 1))]],
        checkbox=("Show overlay", cb),
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert "checkbox" in panel._widgets
        assert isinstance(panel._widgets["checkbox"], mwidgets.CheckButtons)
        panel._widgets["checkbox"].set_active(0)
        assert True in captured
    finally:
        plt.close(fig)
