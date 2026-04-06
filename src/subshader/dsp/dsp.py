"""
DSP Abstract Base Class for SubShader.

Defines the interface that all time-frequency analysis backends must implement.
Each backend follows a three-stage pipeline: pre-process, transform, post-process.
"""

from abc import ABC, abstractmethod

from subshader.config import PipelineConfig


class DSP(ABC):
    """Abstract base for all DSP backends.

    Subclasses implement pre(), transform(), post() for their specific
    time-frequency algorithm. The process() method orchestrates all three.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def process(self, chunk):
        """Run the full pre -> transform -> post pipeline.

        Args:
            chunk: Raw audio samples from AudioStream.

        Returns:
            Processed time-frequency output suitable for visualization.
        """
        data = self.pre(chunk)
        raw = self.transform(data)
        return self.post(raw)

    @abstractmethod
    def pre(self, chunk):
        """Pre-process the input chunk before transform.

        Args:
            chunk: Raw audio samples.

        Returns:
            Validated or pre-processed data.
        """
        ...

    @abstractmethod
    def transform(self, data):
        """Apply the core time-frequency transform.

        Args:
            data: Pre-processed audio data from pre().

        Returns:
            Raw transform output (e.g., complex CWT coefficients).
        """
        ...

    @abstractmethod
    def post(self, raw):
        """Post-process raw transform output.

        Args:
            raw: Output from transform().

        Returns:
            Final processed array ready for visualization.
        """
        ...
