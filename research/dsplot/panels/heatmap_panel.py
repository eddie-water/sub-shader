"""HeatmapPanel — semantic StaticPanel subclass for 2D field displays.

Default (1, 1) is square; callers pass `units=(N, 1)` for wide spectrograms.
"""
from __future__ import annotations

from typing import ClassVar, Tuple

from .static_panel import StaticPanel


class HeatmapPanel(StaticPanel):
    default_units: ClassVar[Tuple[int, int]] = (1, 1)
