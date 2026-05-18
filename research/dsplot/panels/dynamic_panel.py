"""DynamicPanel — multi-frame Panel rendered as a matplotlib FuncAnimation loop.

Frame model: each frame is a `list[Plottable]` drawn onto the panel's axes for
that frame. Frames are either pre-computed (`frames`) or generated lazily
(`frame_fn` + `num_frames`).

LOAD-BEARING DETAIL — DO NOT REMOVE:
    The FuncAnimation instance MUST be held on the Panel via `self._anim`.
    If you write `_ = FuncAnimation(...)` (or otherwise let it fall out of
    scope), matplotlib garbage-collects the animation timer thread and the
    figure silently freezes on frame 0. Test 4
    (`test_dynamic_panel_render_holds_anim_reference`) is the regression
    guard for this assignment.

Per-frame artist cleanup uses tracked-artist removal, NOT `ax.cla()`:
    `ax.cla()` would also nuke the background, spines, and base_plottables,
    forcing them to re-render every frame and dropping FPS. Instead, before
    each frame draws, the artists added by the PREVIOUS frame are removed
    via their tracked `.remove()` handles. Base layers (set up by
    `_render_background`) and base_plottables drawn once during render() are
    never disturbed. Test 9
    (`test_dynamic_panel_base_plottables_survive_frame_transitions`) is the
    regression guard.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

from .. import style
from ..axes_setup import setup_vector_axes
from .base import Panel
from .static_panel import StaticPanel  # for type-cross-ref (not subclassed)


Frame = List["Plottable"]  # noqa: F821 — forward reference for typing only


class DynamicPanel(Panel):
    """Multi-frame Panel rendered as a matplotlib FuncAnimation loop.

    Either `frames` (pre-computed list of frames) or `frame_fn` + `num_frames`
    must be provided. `repeat=True` (default) loops with auto-reset.
    """

    def __init__(
        self,
        *,
        frames: Optional[List[Frame]] = None,
        frame_fn: Optional[Callable[[int], Frame]] = None,
        num_frames: Optional[int] = None,
        interval_ms: int = 250,
        repeat: bool = True,
        base_plottables: Optional[List["Plottable"]] = None,  # noqa: F821
        # standard StaticPanel kwargs forwarded:
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        lim: Optional[Union[float, Tuple[float, float]]] = None,
        axis_style: str = "line",
        axis_labels: bool = False,
        show_border: bool = True,
    ) -> None:
        super().__init__()
        if frames is None and frame_fn is None:
            raise ValueError(
                "DynamicPanel requires either `frames` or `frame_fn` to be set"
            )
        if frame_fn is not None and frames is None and num_frames is None:
            raise ValueError(
                "DynamicPanel(frame_fn=...) requires `num_frames`"
            )

        self.frames = frames
        self.frame_fn = frame_fn
        self.num_frames = num_frames
        self.interval_ms = interval_ms
        self.repeat = repeat
        self.base_plottables = base_plottables or []

        self.title = title
        self.subtitle = subtitle
        self.lim = lim
        self.axis_style = axis_style
        self.axis_labels = axis_labels
        self.show_border = show_border

        self._anim = None
        self._frame_artists: List = []
        # When a Figure orchestrates multiple DynamicPanels, it owns the
        # master FuncAnimation and drives each panel via tick(global_tick).
        # Figure sets this flag before calling render() so we skip our own
        # FuncAnimation construction; the figure's master timer keeps every
        # panel in lockstep.
        self._managed_externally: bool = False
        self._last_drawn_frame: int = -1

    def _scalar_lim(self) -> Optional[float]:
        if self.lim is None:
            return None
        if isinstance(self.lim, tuple):
            lo, hi = self.lim
            return max(abs(lo), abs(hi))
        return float(self.lim)

    def _total_frames(self) -> int:
        if self.frames is not None:
            return len(self.frames)
        return int(self.num_frames)

    def _resolve_frame(self, frame_idx: int) -> Frame:
        if self.frame_fn is not None:
            return self.frame_fn(frame_idx)
        return self.frames[frame_idx]

    def _render_background(self) -> None:
        """Apply panel chrome (bg, spines, axes setup, titles) ONCE.

        Drawn before any frame; never re-drawn during animation. base_plottables
        are drawn on top of the chrome and also persist across frames.
        """
        ax = self.ax
        ax.set_facecolor(style.BG_COLOR)
        if self.show_border:
            for spine in ax.spines.values():
                spine.set_edgecolor(style.SPINE_COLOR)
                spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        setup_vector_axes(
            ax,
            lim=self._scalar_lim(),
            panel_title=None,
            show_border=self.show_border,
            axis_style=self.axis_style,
            axis_labels=self.axis_labels,
        )

        self._render_chrome_titles()

        for plottable in self.base_plottables:
            plottable.draw(ax)

    def _animate(self, frame_idx: int) -> None:
        """Clear previous frame's artists, draw frame_idx's plottables.

        The tracked-artist removal pattern (NOT ax.cla()) preserves the
        background, spines, axes-setup output, and base_plottables across
        frame transitions.
        """
        ax = self.ax
        # Remove artists from previous frame
        for artist in self._frame_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                # Already detached or container artist; ignore.
                pass
        self._frame_artists = []

        # Snapshot artist counts BEFORE this frame's plottables draw
        before_patches = set(id(p) for p in ax.patches)
        before_lines = set(id(p) for p in ax.lines)
        before_collections = set(id(p) for p in ax.collections)
        before_texts = set(id(p) for p in ax.texts)
        before_images = set(id(p) for p in ax.images)

        # Draw frame plottables
        plottables = self._resolve_frame(frame_idx)
        for plottable in plottables:
            plottable.draw(ax)

        # Capture artists added by this frame (by id-diff against the snapshot)
        for collection_name, before_ids in (
            ("patches", before_patches),
            ("lines", before_lines),
            ("collections", before_collections),
            ("texts", before_texts),
            ("images", before_images),
        ):
            for artist in getattr(ax, collection_name):
                if id(artist) not in before_ids:
                    self._frame_artists.append(artist)

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("DynamicPanel.render() called before attach()")

        self._render_background()
        self._frame_artists = []
        self._animate(0)
        self._last_drawn_frame = 0

        # When a Figure is orchestrating this panel (e.g. multi-panel mixed
        # Figure), it installs its own master FuncAnimation that ticks every
        # DynamicPanel from a shared clock. Skip the per-panel FuncAnimation
        # in that case — the figure-level timer is the single source of truth.
        if self._managed_externally:
            return

        # Standalone path: hold the FuncAnimation reference on self so the
        # timer thread isn't garbage-collected when render() returns.
        # Stripping this assignment freezes the animation on frame 0 with no
        # error.
        from matplotlib.animation import FuncAnimation
        self._anim = FuncAnimation(
            self.ax.figure,
            self._animate,
            frames=self._total_frames(),
            interval=self.interval_ms,
            repeat=self.repeat,
            blit=False,
        )

    def tick(self, global_tick: int) -> None:
        """Render the frame indexed by ``global_tick`` on the figure clock.

        Clamped to the panel's last frame: if the figure clock ticks past
        ``total_frames - 1``, the panel holds its final frame. No-ops when the
        target frame is already on screen so trailing holds don't churn
        artists.
        """
        target = min(global_tick, self._total_frames() - 1)
        if target == self._last_drawn_frame:
            return
        self._animate(target)
        self._last_drawn_frame = target

    def save_gif(self, path: str, *, fps: int = 4, dpi: Optional[int] = None) -> str:
        """Save the animation as a GIF via PillowWriter. Returns the path."""
        if self._anim is None:
            raise RuntimeError(
                "DynamicPanel.save_gif() requires render() to be called first"
            )
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        self._anim.save(path, writer=writer, dpi=dpi or style.DEFAULT_DPI)
        return path
