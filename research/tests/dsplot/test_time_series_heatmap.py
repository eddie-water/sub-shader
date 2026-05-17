"""Tests for TimeSeries + Heatmap Plottables and freq_axis helper.

Each behavior in the plan maps to one test:

  1. compute_freq_yticks default ticks (4 labels, "20" / "200" / "2k" / "20k")
  2. compute_freq_yticks custom tick_freqs (3 ticks)
  3. TimeSeries.draw adds a fill_between collection
  4. TimeSeries default color resolves to style.NEUTRAL_COLOR
  5. Heatmap.draw adds an imshow image
  6. Heatmap with log_freq=True sets y-ticks via compute_freq_yticks
  7. Heatmap vmax_percentile resolves vmax to np.percentile(arr, p)
  8. D-05 lazy lookup: reassigning style.DEFAULT_HEATMAP_CMAP between construction
     and draw is observable on the resulting artist.

Test 8 uses try/finally to restore the style default — global reassignment must
not leak across tests.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dsplot
from dsplot import Heatmap, TimeSeries, freq_axis, style


# ============================================================
# freq_axis.compute_freq_yticks
# ============================================================

def test_compute_freq_yticks_default_returns_four_log_octave_labels() -> None:
    freqs = np.geomspace(20, 20000, 100)
    bins, labels = freq_axis.compute_freq_yticks(freqs)

    assert labels == ["20", "200", "2k", "20k"]
    assert len(bins) == 4
    assert all(0 <= b <= len(freqs) for b in bins)


def test_compute_freq_yticks_custom_tick_freqs_returns_three_ticks() -> None:
    freqs = np.geomspace(100, 1000, 50)
    bins, labels = freq_axis.compute_freq_yticks(freqs, tick_freqs=(100, 500, 1000))

    assert len(bins) == 3
    assert len(labels) == 3


# ============================================================
# TimeSeries
# ============================================================

def test_time_series_draw_adds_fill_between_collection() -> None:
    fig, ax = plt.subplots()
    sig = np.sin(np.linspace(0, 2 * np.pi, 100))

    TimeSeries(sig, sample_rate=100).draw(ax)

    assert len(ax.collections) >= 1
    plt.close(fig)


def test_time_series_default_color_resolves_to_neutral() -> None:
    fig, ax = plt.subplots()
    sig = np.sin(np.linspace(0, 2 * np.pi, 100))

    TimeSeries(sig, sample_rate=100).draw(ax)

    # fill_between produces a PolyCollection; default color must be NEUTRAL_COLOR
    # (the dsplot palette reserves PRIMARY/SECONDARY for vector identity; bare
    # TimeSeries is non-identity data).
    facecolor = ax.collections[0].get_facecolor()[0]
    expected = matplotlib.colors.to_rgba(style.NEUTRAL_COLOR)
    assert facecolor == pytest.approx(expected, abs=1e-6)
    plt.close(fig)


# ============================================================
# Heatmap
# ============================================================

def test_heatmap_draw_adds_imshow_image() -> None:
    fig, ax = plt.subplots()
    arr = np.random.RandomState(0).rand(10, 20)

    Heatmap(arr).draw(ax)

    assert len(ax.images) == 1
    plt.close(fig)


def test_heatmap_log_freq_sets_yticks_from_freq_axis() -> None:
    fig, ax = plt.subplots()
    arr = np.random.RandomState(1).rand(10, 20)
    freqs = np.geomspace(20, 20000, 10)

    Heatmap(arr, log_freq=True, freqs=freqs, duration_s=1.0).draw(ax)

    expected_bins, expected_labels = freq_axis.compute_freq_yticks(freqs)
    actual_yticks = list(ax.get_yticks())
    actual_labels = [t.get_text() for t in ax.get_yticklabels()]

    assert actual_yticks == pytest.approx(expected_bins, abs=1e-9)
    assert actual_labels == expected_labels
    plt.close(fig)


def test_heatmap_vmax_percentile_resolves_to_np_percentile() -> None:
    fig, ax = plt.subplots()
    arr = np.linspace(0.0, 1.0, 200).reshape(10, 20)

    Heatmap(arr, vmax_percentile=90.0).draw(ax)

    expected_vmax = float(np.percentile(arr, 90.0))
    actual_vmax = ax.images[0].get_clim()[1]
    assert actual_vmax == pytest.approx(expected_vmax, abs=1e-9)
    plt.close(fig)


def test_heatmap_lazy_lookup_default_cmap_observable() -> None:
    """D-05: reassigning style.DEFAULT_HEATMAP_CMAP between construct and draw
    must be visible on the resulting image's cmap."""
    original = dsplot.style.DEFAULT_HEATMAP_CMAP
    try:
        dsplot.style.DEFAULT_HEATMAP_CMAP = "viridis"
        fig, ax = plt.subplots()
        arr = np.random.RandomState(2).rand(10, 20)

        Heatmap(arr).draw(ax)

        assert ax.images[0].get_cmap().name == "viridis"
        plt.close(fig)
    finally:
        dsplot.style.DEFAULT_HEATMAP_CMAP = original
