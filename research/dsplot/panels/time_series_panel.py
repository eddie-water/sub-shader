"""TimeSeriesPanel — semantic StaticPanel subclass that carries 3-unit width.

Identical chrome to StaticPanel; the only override is `default_units = (3, 1)`
so wide signal/time displays compose naturally in a Figure.compose row.
"""
from __future__ import annotations

from typing import ClassVar, Tuple

from .static_panel import StaticPanel


class TimeSeriesPanel(StaticPanel):
    default_units: ClassVar[Tuple[int, int]] = (3, 1)
