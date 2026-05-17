"""Figure renderers — one module per DSP.md figure.

Each module exposes a ``render(...)`` function returning the absolute output
path. The ``__main__`` dispatcher in this subpackage regenerates every figure
into ``assets/images/dsp/`` in one shot.

Per D-01 (09-CONTEXT.md), this subpackage is **consumer code**: it is
allowed to bridge ``research.utilities`` for CWT computation. The library
isolation grep excludes this directory:
    grep -rIn "subshader" research/dsplot/ --exclude-dir=figures
"""
