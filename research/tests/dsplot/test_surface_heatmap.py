"""Tests for the SurfaceHeatmap Plottable.

  1. SurfaceHeatmap.draw adds a Poly3DCollection to ax.collections when
     drawn onto an Axes3D.
  2. vmax resolution is shared with Heatmap (_resolve_vmax honors the same
     explicit-vmax -> vmax_percentile -> style default priority).
  3. normalize_to remaps the surface into the [-n, +n] / [0, n] cube used by
     StaticPanel3D.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dsplot import SurfaceHeatmap


def _axes3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return fig, ax


def test_surface_heatmap_draw_adds_poly3dcollection() -> None:
    fig, ax = _axes3d()
    arr = np.random.RandomState(0).rand(10, 20)

    SurfaceHeatmap(arr).draw(ax)

    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    plt.close(fig)


def test_surface_heatmap_vmax_resolution_matches_heatmap() -> None:
    arr = np.linspace(0.0, 1.0, 200).reshape(10, 20)

    surface = SurfaceHeatmap(arr, vmax_percentile=90.0)

    expected_vmax = float(np.percentile(arr, 90.0))
    assert surface._resolve_vmax() == pytest.approx(expected_vmax, abs=1e-9)


def test_surface_heatmap_normalize_to_remaps_into_symmetric_cube() -> None:
    # Axes3D autoscales its data limits to the plotted surface (with a small
    # margin), so the post-draw xlim/ylim/zlim double as a proxy for the
    # remapped vertex bounds without reaching into Poly3DCollection internals.
    fig, ax = _axes3d()
    arr = np.linspace(0.0, 1.0, 200).reshape(10, 20)
    lim = 1.0

    SurfaceHeatmap(arr, vmax=1.0, normalize_to=lim).draw(ax)

    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    assert max(abs(xlim[0]), abs(xlim[1])) == pytest.approx(lim, rel=0.2)
    assert max(abs(ylim[0]), abs(ylim[1])) == pytest.approx(lim, rel=0.2)
    assert zlim[0] >= -0.1 * lim
    assert zlim[1] == pytest.approx(lim, rel=0.2)
    plt.close(fig)
