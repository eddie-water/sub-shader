"""DynamicPanel tests — Plan 09-04 Task 1.

Verifies the 9 behaviors from the plan:
  1. constructs with frames list
  2. ValueError when neither frames nor frame_fn set
  3. ValueError when frame_fn set without num_frames
  4. attach+render sets up FuncAnimation reference at panel._anim
  5. panel._anim has event_source + repeat flag honored
  6. _animate(0) draws frame-0 plottables
  7. _animate(1) then _animate(0) leaves same artist counts as _animate(0)
     alone — proves per-frame artist cleanup works
  8. save_gif writes a file > 0 bytes
  9. base_plottables drawn once; survive frame transitions (axhline/axvline
     count stable across multiple _animate calls)

`pytest.importorskip("dsplot", ...)` keeps the suite collectable when 09-01
hasn't shipped yet (it has, but this matches the project convention).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


pytest.importorskip("dsplot", reason="dsplot package not yet available")


def _fresh_axes():
    fig, ax = plt.subplots()
    return fig, ax


def test_dynamic_panel_constructs_with_frames_list():
    from dsplot import DynamicPanel, Vector
    panel = DynamicPanel(frames=[[Vector((1, 0))], [Vector((0, 1))]])
    assert panel is not None


def test_dynamic_panel_requires_frames_or_frame_fn():
    from dsplot import DynamicPanel
    with pytest.raises(ValueError):
        DynamicPanel()


def test_dynamic_panel_frame_fn_requires_num_frames():
    from dsplot import DynamicPanel, Vector
    with pytest.raises(ValueError):
        DynamicPanel(frame_fn=lambda i: [Vector((i, 0))])


def test_dynamic_panel_render_holds_anim_reference():
    from dsplot import DynamicPanel, Vector
    panel = DynamicPanel(frames=[[Vector((1, 0))], [Vector((0, 1))]])
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        assert panel._anim is not None
        from matplotlib.animation import FuncAnimation
        assert isinstance(panel._anim, FuncAnimation)
    finally:
        plt.close(fig)


def test_dynamic_panel_anim_repeat_flag_honored():
    from dsplot import DynamicPanel, Vector
    panel = DynamicPanel(
        frames=[[Vector((1, 0))], [Vector((0, 1))]],
        repeat=True,
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        # FuncAnimation stores the repeat flag as ._repeat (private but stable)
        assert panel._anim._repeat is True
        # event_source should be a real timer (or None depending on backend);
        # the load-bearing assertion is that the attribute exists.
        assert hasattr(panel._anim, "event_source")
    finally:
        plt.close(fig)


def test_dynamic_panel_animate_zero_draws_frame_zero_plottables():
    from dsplot import DynamicPanel, Vector
    frames = [
        [Vector((1, 0), label="f0")],
        [Vector((0, 1), label="f1")],
    ]
    panel = DynamicPanel(frames=frames)
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        # render() called _animate(0); inspect text artists for the frame-0 label
        labels = [t.get_text() for t in ax.texts]
        assert "f0" in labels
        # frame 1 label should NOT appear yet
        assert "f1" not in labels
    finally:
        plt.close(fig)


def test_dynamic_panel_animate_cleans_previous_frame_artists():
    """_animate(1) then _animate(0) → same artist count as _animate(0) alone.

    Proves the tracked-artist removal pattern works: no accumulation across
    frames.
    """
    from dsplot import DynamicPanel, Vector
    frames = [
        [Vector((1, 0), label="f0")],
        [Vector((0, 1), label="f1")],
    ]
    panel_a = DynamicPanel(frames=frames)
    fig_a, ax_a = _fresh_axes()
    panel_b = DynamicPanel(frames=frames)
    fig_b, ax_b = _fresh_axes()
    try:
        panel_a.attach(ax_a)
        panel_a.render()  # ends with _animate(0)
        baseline_patches = len(ax_a.patches)
        baseline_texts = len([t for t in ax_a.texts if t.get_text() in ("f0", "f1")])

        panel_b.attach(ax_b)
        panel_b.render()  # ends with _animate(0)
        panel_b._animate(1)  # advance
        panel_b._animate(0)  # back to frame 0
        cycled_patches = len(ax_b.patches)
        cycled_texts = len([t for t in ax_b.texts if t.get_text() in ("f0", "f1")])

        assert cycled_patches == baseline_patches, (
            f"per-frame artist cleanup failed: "
            f"baseline patches={baseline_patches}, cycled patches={cycled_patches}"
        )
        assert cycled_texts == baseline_texts, (
            f"per-frame text cleanup failed: "
            f"baseline texts={baseline_texts}, cycled texts={cycled_texts}"
        )
    finally:
        plt.close(fig_a)
        plt.close(fig_b)


def test_dynamic_panel_save_gif_writes_nonempty_file(tmp_path):
    pytest.importorskip("PIL")
    from dsplot import DynamicPanel, Vector
    panel = DynamicPanel(
        frames=[[Vector((1, 0))], [Vector((0, 1))]],
        interval_ms=100,
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()
        out_path = tmp_path / "out.gif"
        result = panel.save_gif(str(out_path), fps=4)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert isinstance(result, str)
    finally:
        plt.close(fig)


def test_dynamic_panel_base_plottables_survive_frame_transitions():
    """base_plottables draw once during render(); _animate calls only clear
    the frame-varying artists, not the base layer.

    Uses a simple fake "axline" base plottable that adds one axhline + one
    axvline. After multiple _animate calls, axhline+axvline count must stay
    constant.
    """
    from dsplot import DynamicPanel, Vector
    from dsplot.plottables.base import Plottable

    class _AxLinePair(Plottable):
        def draw(self, ax):
            ax.axhline(0, color="white", linewidth=0.5)
            ax.axvline(0, color="white", linewidth=0.5)

    panel = DynamicPanel(
        frames=[
            [Vector((1, 0), label="f0")],
            [Vector((0, 1), label="f1")],
            [Vector((-1, 0), label="f2")],
        ],
        base_plottables=[_AxLinePair()],
    )
    fig, ax = _fresh_axes()
    try:
        panel.attach(ax)
        panel.render()  # base drawn once; _animate(0) called
        initial_lines = len(ax.lines)
        panel._animate(1)
        after_one = len(ax.lines)
        panel._animate(2)
        after_two = len(ax.lines)
        panel._animate(0)
        back_to_zero = len(ax.lines)
        assert initial_lines == after_one == after_two == back_to_zero, (
            f"base_plottable line count drifted across frames: "
            f"{initial_lines}, {after_one}, {after_two}, {back_to_zero}"
        )
        assert initial_lines >= 2, "expected at least 2 base lines (axhline + axvline)"
    finally:
        plt.close(fig)
