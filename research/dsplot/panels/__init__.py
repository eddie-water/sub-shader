"""Panel containers — own one mpl Axes and a list of Plottables."""
from .base import Panel
from .composite_panel import CompositePanel
from .dynamic_panel import DynamicPanel
from .heatmap_panel import HeatmapPanel
from .interactive_panel import InteractivePanel
from .static_panel import StaticPanel
from .static_panel_3d import StaticPanel3D
from .suptitle_panel import SuptitlePanel
from .text_panel import TextPanel
from .time_series_panel import TimeSeriesPanel

__all__ = [
    "Panel",
    "StaticPanel",
    "StaticPanel3D",
    "DynamicPanel",
    "InteractivePanel",
    "TimeSeriesPanel",
    "HeatmapPanel",
    "CompositePanel",
    "TextPanel",
    "SuptitlePanel",
]
