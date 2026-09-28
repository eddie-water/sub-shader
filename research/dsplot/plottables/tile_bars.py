"""TileBars Plottable - 2D array rendered as 3D boxes on a resolution lattice.

Sibling of `SurfaceHeatmap`: the same (n_freq_rows x n_time_cols) magnitude
array, but instead of a continuous terrain each box's FOOTPRINT is one
analysis tile - a band's time-frequency resolution cell - and its height is
the mean magnitude pooled inside that tile. An STFT bank renders as uniform
boxes; a constant-Q CWT bank renders boxes whose time-width shrinks as the
band frequency rises. Near-floor tiles stay visible as flat lattice cells,
so unactivated bands read as "potential" and energy reads as activation.

Subclasses `Heatmap` so vmin/vmax/cmap/extent/duration_s/freqs behave
exactly as they do for the 2D panels and for `SurfaceHeatmap`; the
`normalize_to`/`z_base`/`gamma` cube-mapping knobs mirror `SurfaceHeatmap`.

Tile geometry is caller-supplied, in the units of the data grid:
  - `row_edges`: tile boundaries along the frequency axis, in ROW-BIN units
    (floats, ascending, within [0, n_rows]). One tile row per consecutive
    pair. On a log-spaced grid, one edge per row gives constant-Q tiles;
    edges converted from linear-Hz bin boundaries give STFT tiles.
  - `tile_seconds`: time-width of the tiles in each tile row - scalar
    (uniform bank) or per-tile-row array (constant-Q: num_cycles / f).
  - `max_cols_per_row` clamps how many boxes a row may split into; rows
    whose native tile is finer pool up to the clamp (sub-pixel boxes would
    alias anyway and explode the polygon count).
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import style
from .heatmap import Heatmap


class TileBars(Heatmap):
    """Resolution-lattice box plot for an Axes3D (e.g. `StaticPanel3D`)."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        row_edges: np.ndarray,
        tile_seconds: float | np.ndarray,
        max_cols_per_row: int = 100,
        pool: str = "mean",
        normalize_to: float | None = None,
        z_base: float = 0.0,
        gamma: float | None = None,
        floor_height_frac: float = 0.01,
        bar_alpha: float = 1.0,
        edge_color: str | None = None,
        shade: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)
        self.row_edges = np.asarray(row_edges, dtype=np.float64)
        n_tiles = len(self.row_edges) - 1
        self.tile_seconds = np.broadcast_to(
            np.asarray(tile_seconds, dtype=np.float64), (n_tiles,)
        ).copy()
        self.max_cols_per_row = int(max_cols_per_row)
        # Tile pooling domain. "mean": plain arithmetic mean of the data
        # values. "db_energy": treat values as dB magnitudes - average the
        # POWER (10^(dB/10)) inside each tile and convert back to dB, so a
        # tile's height reports its actual energy content and matches the
        # 2D dB heatmap's scaling instead of a log-domain mean (which
        # underweights peaks). "db_peak": max dB inside each tile - matches
        # the per-pixel peak reading of a 2D scalogram, so brief transients
        # (clicks) stay bright instead of being diluted by the tile's dwell
        # time the way an energy mean dilutes them.
        if pool not in ("mean", "db_energy", "db_peak"):
            raise ValueError(
                f"pool must be 'mean', 'db_energy' or 'db_peak', got {pool!r}")
        self.pool = pool
        self.normalize_to = normalize_to
        self.z_base = float(z_base)
        self.gamma = float(gamma) if gamma is not None else None
        # Minimum box height as a fraction of the cube's z span - keeps
        # silent tiles visible as a flat lattice instead of vanishing.
        self.floor_height_frac = float(floor_height_frac)
        self.bar_alpha = float(bar_alpha)
        self.edge_color = edge_color
        self.shade = bool(shade)

    def draw(self, ax: Axes) -> None:
        cmap = self.cmap if self.cmap is not None else style.DEFAULT_HEATMAP_CMAP
        import matplotlib.pyplot as plt
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        vmax = self._resolve_vmax()

        if self.extent is not None:
            x0, x1, y0, y1 = self.extent
        elif self.duration_s is not None and self.freqs is not None:
            x0, x1, y0, y1 = 0.0, float(self.duration_s), 0.0, float(len(self.freqs))
        else:
            h, w = self.data.shape
            x0, x1, y0, y1 = 0.0, float(w), 0.0, float(h)

        data = np.asarray(self.data, dtype=np.float64)
        if self.pool == "db_energy":
            data = 10.0 ** (data / 10.0)   # dB -> power for tile averaging
        n_rows, n_cols = data.shape
        duration = x1 - x0

        # Pool each tile row (a slice of data rows) into its per-band time
        # windows: one box per (tile row, time window), height = mean.
        xs, ys, dxs, dys, heights = [], [], [], [], []
        for i in range(len(self.row_edges) - 1):
            r_lo = int(np.floor(self.row_edges[i]))
            r_hi = max(r_lo + 1, int(np.ceil(self.row_edges[i + 1])))
            r_lo = max(0, min(r_lo, n_rows - 1))
            r_hi = max(r_lo + 1, min(r_hi, n_rows))
            if self.pool == "db_peak":
                band = data[r_lo:r_hi, :].max(axis=0)
            else:
                band = data[r_lo:r_hi, :].mean(axis=0)

            n_tiles_t = max(1, int(round(duration / self.tile_seconds[i])))
            n_tiles_t = min(n_tiles_t, self.max_cols_per_row, n_cols)
            col_edges = np.linspace(0, n_cols, n_tiles_t + 1).astype(int)
            if self.pool == "db_peak":
                means = np.maximum.reduceat(band, col_edges[:-1])
            else:
                sums = np.add.reduceat(band, col_edges[:-1])
                counts = np.diff(col_edges).astype(np.float64)
                means = sums / counts

            t_edges = x0 + (col_edges / n_cols) * duration
            y_lo = y0 + (self.row_edges[i] / n_rows) * (y1 - y0)
            y_hi = y0 + (self.row_edges[i + 1] / n_rows) * (y1 - y0)
            xs.append(t_edges[:-1])
            dxs.append(np.diff(t_edges))
            ys.append(np.full(n_tiles_t, y_lo))
            dys.append(np.full(n_tiles_t, y_hi - y_lo))
            heights.append(means)

        x = np.concatenate(xs)
        dx = np.concatenate(dxs)
        y = np.concatenate(ys)
        dy = np.concatenate(dys)
        mag = np.concatenate(heights)
        if self.pool == "db_energy":
            mag = 10.0 * np.log10(np.maximum(mag, 1e-12))  # power -> dB

        span = max(vmax - self.vmin, 1e-12)
        m01 = np.clip((mag - self.vmin) / span, 0.0, 1.0)
        if self.gamma is not None:
            m01 = m01 ** self.gamma

        if self.normalize_to is not None:
            n = float(self.normalize_to)
            z_top = n
            x = np.interp(x, (x0, x1), (-n, n))
            dx = dx * (2.0 * n / (x1 - x0))
            y = np.interp(y, (y0, y1), (-n, n))
            dy = dy * (2.0 * n / (y1 - y0))
        else:
            z_top = float(vmax)
        z_span = z_top - self.z_base
        dz = np.maximum(m01 * z_span, self.floor_height_frac * z_span)
        z = np.full_like(dz, self.z_base)

        colors = cmap(m01)
        ax.bar3d(
            x, y, z, dx, dy, dz,
            color=colors,
            edgecolor=self.edge_color,
            linewidth=0.3 if self.edge_color is not None else 0.0,
            alpha=self.bar_alpha,
            shade=self.shade,
            zsort="average",
            zorder=self.zorder,
        )
