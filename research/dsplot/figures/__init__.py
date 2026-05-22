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
"""
from .components_recombine import show as figure_2_4_1
from .figure_1 import show_hero as figure_1

__all__ = [
    "figure_1",
    "figure_2_4_1",
]
