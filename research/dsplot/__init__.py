"""dsplot — standalone matplotlib-based plotting library.

Two layers: **Plottables** (units drawn onto a matplotlib Axes) and **Panels**
(containers that own an Axes and orchestrate static / animated / interactive
rendering).

`dsplot.style` is the inheritable template every figure inherits from. Two
override modes are supported:

  1. Global reassignment:
         import dsplot
         dsplot.style.PRIMARY_COLOR = "#new"
     Affects every figure rendered after.

  2. Local override (a figure module derives its own constant from a default):
         from dsplot import style
         LABEL_RATIO = style.DEFAULT_LABEL_RATIO * 1.5
     Affects only that figure; the global default is untouched.

Plottables resolve None-valued style knobs against `dsplot.style.*` at draw()
time (lazy lookup), so a global reassignment between construction and draw
uses the NEW value.
"""

from . import freq_axis, style
from .figure import Figure, apply_jupyter_dark
from .panels import (
    CompositePanel,
    DynamicPanel,
    HeatmapPanel,
    InteractivePanel,
    Panel,
    StaticPanel,
    StaticPanel3D,
    TimeSeriesPanel,
)
from .plottables import (
    Annotation,
    Dropline,
    Heatmap,
    Line,
    Spotlight,
    TimeSeries,
    Vector,
    VectorComponents,
)

__all__ = [
    "style",
    "freq_axis",
    "Vector",
    "VectorComponents",
    "Annotation",
    "TimeSeries",
    "Heatmap",
    "Line",
    "Spotlight",
    "Dropline",
    "Panel",
    "StaticPanel",
    "StaticPanel3D",
    "DynamicPanel",
    "InteractivePanel",
    "TimeSeriesPanel",
    "HeatmapPanel",
    "CompositePanel",
    "Figure",
    "apply_jupyter_dark",
]
