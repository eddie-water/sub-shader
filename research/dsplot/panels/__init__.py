"""Panel containers — own one mpl Axes and a list of Plottables."""
from .base import Panel
from .dynamic_panel import DynamicPanel
from .interactive_panel import InteractivePanel
from .static_panel import StaticPanel

__all__ = ["Panel", "StaticPanel", "DynamicPanel", "InteractivePanel"]
