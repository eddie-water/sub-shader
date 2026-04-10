"""Intensity tracking for consistent colormap scaling across frames."""

import numpy as np
from typing import Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class IntensityTracker:
    """Tracks global max intensity across frames for consistent colormap scaling."""

    def __init__(
        self,
        percentile: float = 99.0,
        retention_rate: float = 0.95,
        floor_value: float = 1e-8,
        warmup_frames: int = 10,
    ):
        """
        Initialize the intensity tracker.

        Args:
            percentile: Percentile of frame data used as frame_max (robust to spike values).
            retention_rate: Fraction of global_max retained each frame (0.95 = retain 95%).
                A value of 0.95 means global_max decays slowly on quiet passages while
                new frame peaks can raise it immediately.
            floor_value: Minimum value for global_max (prevents division-by-zero in shader).
            warmup_frames: Number of frames before tracker is considered ready.
        """
        self.percentile = percentile
        self.retention_rate = retention_rate
        self.floor_value = floor_value
        self.warmup_frames = warmup_frames

        self.global_max = 0.0
        self.frame_count = 0
        self.is_ready = False

    def update(self, frame: Union[np.ndarray, 'cp.ndarray']) -> float:
        """
        Update the global max with a new frame.

        Args:
            frame: The frame to update the global max with.

        Returns:
            The current global max value (for use as vmax in colormap).
        """
        flat_data = frame.flatten()
        frame_max = float(np.percentile(flat_data, self.percentile))

        # Slow decay allows max to decrease over time if intensity drops
        self.global_max = self.retention_rate * self.global_max
        self.global_max = max(self.global_max, self.floor_value)
        self.global_max = max(self.global_max, frame_max)

        self.frame_count += 1
        if self.frame_count >= self.warmup_frames:
            self.is_ready = True

        return self.global_max

    def reset(self) -> None:
        """Reset the tracker state."""
        self.global_max = 0.0
        self.frame_count = 0
        self.is_ready = False
