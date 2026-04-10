"""Intensity reference for consistent colormap scaling across frames.

IntensityTracker holds a fixed normalization reference computed once before
the render loop starts (via pipeline pre-scan). The value never changes
during playback, ensuring the same audio level always maps to the same
pixel brightness regardless of what came before.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class IntensityTracker:
    """Fixed normalization reference for consistent colormap scaling.

    Holds a single fixed intensity max value set at construction. The value
    is clamped to floor_value to prevent division-by-zero in the shader.
    update() is a no-op — the reference never adapts during playback.
    """

    def __init__(
        self,
        fixed_max: float,
        floor_value: float = 1e-8,
    ):
        """
        Initialize the intensity tracker with a fixed reference value.

        Args:
            fixed_max: The normalization reference intensity. Typically the
                result of a pre-scan percentile across the audio file.
            floor_value: Minimum value for global_max (prevents division-by-zero
                in the shader). Clamping is applied if fixed_max < floor_value.
        """
        self.floor_value = floor_value
        self.global_max = max(fixed_max, floor_value)

    def update(self, frame: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        No-op update — returns the fixed global_max unchanged.

        The frame argument is accepted for interface compatibility with
        CircularFrameBuffer, but is not used.

        Args:
            frame: Unused. Present for interface compatibility.

        Returns:
            The fixed global_max value.
        """
        return self.global_max
