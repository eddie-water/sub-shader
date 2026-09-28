"""SurfaceHeatmap Plottable - 2D array rendered as a 3D topographic surface.

Subclasses `Heatmap` so vmax resolution (`_resolve_vmax`, per D-05 lazy
`style.DEFAULT_HEATMAP_CMAP` lookup) and constructor knobs (`vmin`, `vmax`,
`vmax_percentile`, `cmap`, `extent`, `duration_s`, `freqs`) are shared with
the 2D Heatmap - the same CWT array can drive either a flat imshow panel or
a StaticPanel3D surface with zero data-prep changes.

`normalize_to` remaps the surface into a StaticPanel3D's symmetric ±lim
cube: x/y -> `[-normalize_to, +normalize_to]`, z -> `[0, normalize_to]`.
Without it, a `[0, duration_s] x [0, n_freqs] x [0, vmax]` surface lands
outside the panel's cube - StaticPanel3D forces symmetric ±lim_3d on all
three axes (see `panels/static_panel_3d.py::apply_3d_scene`).

`rcount`/`ccount` cap the mesh resolution of the drawn surface. The data is
block-MEAN-pooled down to that grid BEFORE `ax.plot_surface` - passing the
raw array with plot_surface's own rcount/ccount would stride-sample it
(pick every Nth column), which aliases a dense CWT (sample-rate time
columns) into spike noise instead of terrain. Mean pooling is the
anti-aliasing filter; `plot_surface` then draws the pooled grid 1:1.
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import style
from .heatmap import Heatmap


class SurfaceHeatmap(Heatmap):
    """2D array rendered as a 3D surface via `ax.plot_surface`.

    Draws onto an Axes3D (e.g. hosted by `StaticPanel3D`). Overrides only
    `draw(ax)` - every other Heatmap behavior (vmax resolution, extent
    resolution, lazy cmap default) is inherited unchanged.

    `floor_projection=True` additionally drops a flattened `ax.contourf`
    "shadow" of the surface onto the cube floor for a topo-map look.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        rcount: int = 64,
        ccount: int = 160,
        normalize_to: float | None = None,
        z_base: float = 0.0,
        gamma: float | None = None,
        smooth_sigma: float | tuple[float, float] | None = None,
        mesh_upsample: float | tuple[float, float] | None = None,
        floor_projection: bool = False,
        floor_alpha: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)
        self.rcount = int(rcount)
        self.ccount = int(ccount)
        self.normalize_to = normalize_to
        # Cube-coord z of the surface base when normalize_to is set. Default
        # 0.0 keeps the legacy "sits on the floor grid" placement (surface in
        # the cube's upper half). Pass -normalize_to to span the full cube
        # height - fills the viewport vertically when the floor grid is off.
        self.z_base = float(z_base)
        # Power-law compression of the magnitude axis: gamma < 1 lifts
        # low-magnitude features (e.g. a broadband click transient sitting far
        # below a tonal ridge that clips at vmax) into visible relief. Applied
        # to the [vmin, vmax]-normalized magnitude, so height AND color
        # compress together. None = linear (no compression).
        self.gamma = float(gamma) if gamma is not None else None
        # Gaussian blur (in POOLED-grid bins) applied after pooling, before
        # gamma. Scalar = isotropic; (row_sigma, col_sigma) = anisotropic
        # (rows = freq bins, cols = time). Smooths the picket-fence striations
        # a 1-2-bin-wide ridge leaves on the mesh - a render-side fix; the
        # underlying analysis data is untouched.
        self.smooth_sigma = smooth_sigma
        # Cubic-spline upsampling of the pooled mesh (scipy.ndimage.zoom
        # factor; scalar or (row_factor, col_factor)). The anti-terracing
        # fix: a ridge crest landing BETWEEN coarse freq bins renders as
        # linear facet "ribs" - spline interpolation curves through the bin
        # values instead, keeping peaks at full height (unlike smooth_sigma,
        # which low-passes them down). Applied after pooling, before gamma.
        self.mesh_upsample = mesh_upsample
        self.floor_projection = bool(floor_projection)
        self.floor_alpha = float(floor_alpha)

    def draw(self, ax: Axes) -> None:
        cmap = self.cmap if self.cmap is not None else style.DEFAULT_HEATMAP_CMAP
        vmax = self._resolve_vmax()

        if self.extent is not None:
            x0, x1, y0, y1 = self.extent
        elif self.duration_s is not None and self.freqs is not None:
            x0, x1, y0, y1 = 0.0, float(self.duration_s), 0.0, float(len(self.freqs))
        else:
            h, w = self.data.shape
            x0, x1, y0, y1 = 0.0, float(w), 0.0, float(h)

        Z = _pool_mean(self.data, self.rcount, self.ccount)
        if self.smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma=self.smooth_sigma)
        if self.mesh_upsample is not None:
            from scipy.ndimage import zoom as ndimage_zoom
            Z = ndimage_zoom(Z, self.mesh_upsample, order=3, mode="nearest")
        if self.gamma is not None:
            span = max(vmax - self.vmin, 1e-12)
            z01 = np.clip((Z - self.vmin) / span, 0.0, 1.0) ** self.gamma
            Z = self.vmin + z01 * span
        h, w = Z.shape
        X, Y = np.meshgrid(np.linspace(x0, x1, w), np.linspace(y0, y1, h))

        vmin, vmax_plot, floor = self.vmin, vmax, self.vmin
        if self.normalize_to is not None:
            n = float(self.normalize_to)
            X = np.interp(X, (x0, x1), (-n, n))
            Y = np.interp(Y, (y0, y1), (-n, n))
            Z = np.interp(Z, (self.vmin, vmax), (self.z_base, n))
            vmin, vmax_plot, floor = self.z_base, n, self.z_base

        ax.plot_surface(
            X, Y, Z,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_plot,
            rcount=Z.shape[0],
            ccount=Z.shape[1],
            shade=False,
            antialiased=False,
            linewidth=0,
            zorder=self.zorder,
        )

        if self.floor_projection:
            ax.contourf(
                X, Y, Z,
                zdir="z",
                offset=floor,
                cmap=cmap,
                alpha=self.floor_alpha,
            )


def _pool_mean(data: np.ndarray, max_rows: int, max_cols: int) -> np.ndarray:
    """Block-mean pool `data` down to at most (max_rows, max_cols).

    Acts as the anti-aliasing filter for surface rendering: every output
    cell is the mean of its contiguous input block (via `np.add.reduceat`
    over uneven block edges), so no input column is silently skipped the
    way stride sampling would. Axes already at or under the cap pass
    through untouched.
    """
    pooled = np.asarray(data, dtype=np.float64)
    for axis, target in ((0, int(max_rows)), (1, int(max_cols))):
        n = pooled.shape[axis]
        if target <= 0 or n <= target:
            continue
        edges = np.linspace(0, n, target + 1).astype(int)
        sums = np.add.reduceat(pooled, edges[:-1], axis=axis)
        counts = np.diff(edges).astype(np.float64)
        shape = [1, 1]
        shape[axis] = target
        pooled = sums / counts.reshape(shape)
    return pooled
