"""Example unit test — teaching artifact for downstream dsplot consumers.

This file is intentionally minimal. It demonstrates the shape a downstream
test suite (subshader's own tests, an external project's tests, etc.) should
follow when verifying a figure renders correctly:

  1. Import the dsplot pieces you need (top-level re-exports are stable).
  2. Build a Figure with one or more Panels, add Plottables to each panel.
  3. Render and savefig() to a pytest tmp_path so the artifact is cleaned up
     automatically between runs.
  4. Assert the PNG exists and is non-empty.

There is no fixture, no parametrization, no full coverage — that is by design.
This is a SHAPE TEMPLATE, not a coverage suite. The full dsplot coverage lives
in the sibling test files (`test_plottable_construction.py`,
`test_panel_composition.py`, `test_figure_orchestrator.py`, etc.).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — required before any pyplot import in tests


def test_dsplot_example_composition(tmp_path):
    """Smoke-test the canonical Figure + StaticPanel + Vector composition pattern.

    Builds a single-panel vector figure, renders it, saves it to tmp_path, and
    asserts the PNG file exists and is non-empty. This is the minimum a
    downstream consumer test should do to verify dsplot is wired up correctly
    inside their environment.
    """
    from dsplot import Figure, StaticPanel, Vector

    fig = Figure(n_rows=1, n_cols=1, suptitle="dsplot example")
    panel = StaticPanel(
        title="A simple vector",
        lim=2.0,
        axis_style="arrow",
        axis_labels=True,
        show_border=False,
    )
    panel.add(Vector((1.5, 1.0), label="a"))
    fig.add_panel(panel, row=0, col=0)
    fig.render()

    out_path = tmp_path / "example_vector_figure.png"
    fig.savefig(str(out_path))
    fig.close()

    assert out_path.exists(), f"savefig did not produce {out_path}"
    assert out_path.stat().st_size > 0, f"savefig produced an empty file at {out_path}"
