"""Figure renderers — one module per DSP.md figure.

Each module exposes a ``render(...)`` function returning the absolute output
path. The ``__main__`` dispatcher in this subpackage regenerates every figure
into ``assets/images/dsp/figures/<family>/`` in one shot.

Per D-01 (09-CONTEXT.md), this subpackage is **consumer code**: it is
allowed to bridge ``research.utilities`` for CWT computation. The library
isolation grep excludes this directory:
    grep -rIn "subshader" research/dsplot/ --exclude-dir=figures

Notebook re-exports — figures that ship with a ``show()`` for inline
display in dsp.ipynb are re-exported here under their figure-number alias
so a single ``from dsplot.figures import figure_X_Y_Z`` import wires up
every cell's one-line invocation. Add new entries below as new figures
gain a ``show()`` function.

Three-mode invocation contract (convention for every figure module)
====================================================================

Every figure module in this subpackage SHOULD expose three entrypoints
with identical signatures:

    render(output_dir: str, output_filename: str) -> str
        Build, render, and save the figure at production DPI / unit_inches.
        Returns absolute output path. Used by the ``__main__`` batch
        dispatcher and by anything that wants a static PNG on disk.

    show() -> Figure
        Notebook-tuned rendering for dsp.ipynb inline display. Smaller DPI
        (~60) and unit_inches (~2.5) so the figure fits inside a Jupyter
        cell. Suppresses ipympl widget chrome on the returned Figure before
        plt.show() so the canvas reads as pure figure content.

    embed(target: object | None = None) -> Figure
        Drop the figure into a caller-provided matplotlib container.
        ``target=None`` behaves like show() without chrome suppression
        (subshader tests / benchmarks own their own display loop).
        ``target: matplotlib.axes.Axes`` re-hosts a single panel into the
        caller's Axes — only supported by single-panel figure modules.
        ``target: matplotlib.figure.Figure`` re-hosts the whole layout —
        supported only where the layout can adapt cleanly.

Implementation pattern: a private ``_build_figure(unit_inches=None,
dpi=None) -> Figure`` helper that all three modes call with different
sizing knobs. ``sample_template`` is the canonical reference for this
contract.
"""
from .gen_figure_241_xy_recombine_independent import show as figure_2_4_1
from .gen_figure_242_a_onto_b import show as figure_2_4_2
from .gen_figure_243_dot_product_3d import show as figure_2_4_3
from .gen_figure_2_5_sign_accumulation import show as figure_2_5
from .gen_figure_2_6_sine_basis import show as figure_2_6
from .gen_figure_1_stft_vs_cwt import show_hero as figure_1
from . import sample_template

__all__ = [
    "figure_1",
    "figure_2_4_1",
    "figure_2_4_2",
    "figure_2_4_3",
    "figure_2_5",
    "figure_2_6",
    "sample_template",
]
