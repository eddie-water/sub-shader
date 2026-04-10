"""
Frame buffer classes for the renderer module.

CircularFrameBuffer — stores CWT frames in a circular array for scrolling visualization.
AudioFrameBuffer   — stores audio chunks in a circular array for waveform display.
"""

import numpy as np

from subshader.utils.logging import get_logger

log = get_logger(__name__)


class CircularFrameBuffer:
    def __init__(self, frame_shape: tuple[int, int], num_frames: int) -> None:
        """
        Handles circular buffer for scrolling visualization

        Args:
            frame_shape (tuple[int, int]): Shape (height, width) of each frame.
            num_frames (int): Number of frames to store.
        """
        self.num_frames = num_frames
        self.height, self.width = frame_shape

        log.info(f"Plotting {self.num_frames} {frame_shape} sized frames")

        # Store full frames (no overlap)
        self.frames = np.zeros((num_frames, self.height, self.width), dtype=np.float32)
        self.frame_index = 0

        # Pre-allocate flattened buffer
        self.flattened_buffer = np.zeros((self.height, self.width * num_frames), dtype=np.float32)

    # =========================================================================
    # PUBLIC METHODS - External interface
    # =========================================================================

    def push_frame(self, frame_data) -> None:
        """
        Add new frame to circular buffer and update flattened buffer

        Args:
            frame_data (np.ndarray): The new frame data to add to the circular buffer.
        """
        if frame_data.shape != (self.height, self.width):
            log.error(f"Frame data shape mismatch: expected {(self.height, self.width)}, got {frame_data.shape}")
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {frame_data.shape}")

        self.frames[self.frame_index] = frame_data

        # Calculate the correct order of frames (oldest first)
        self.frame_index = (self.frame_index + 1) % self.num_frames
        frame_order = [(self.frame_index + i) % self.num_frames for i in range(self.num_frames)]

        # Populate the flattened buffer with the correct order of frames
        for i, frame_i in enumerate(frame_order):
            start_col = i * self.width
            end_col = start_col + self.width
            self.flattened_buffer[:, start_col:end_col] = self.frames[frame_i]

    def pop_frame(self, index: int) -> np.ndarray:
        """
        Get a specific frame from the circular buffer
        """
        self.frame_index = (self.frame_index - 1) % self.num_frames
        return self.frames[self.frame_index]

    def get_shape(self) -> tuple[int, int]:
        """
        Get the shape of the entire, flattened frame buffer

        Returns:
            tuple: Shape of the flattened buffer.
        """
        return self.flattened_buffer.shape

    def get_flattened_buffer(self) -> np.ndarray:
        """
        Get time-ordered flattened buffer for texture

        Returns:
            np.ndarray: Time-ordered flattened buffer.
        """
        return self.flattened_buffer


class AudioFrameBuffer:
    """
    Circular buffer for 1D audio chunks.

    Stores audio chunks chronologically and outputs a flattened 1D array
    of all samples in time order (oldest first, newest last).
    """

    def __init__(self, chunk_size: int, num_chunks: int) -> None:
        """
        Initialize audio circular buffer.

        Args:
            chunk_size: Number of samples per audio chunk
            num_chunks: Number of chunks to store in the buffer
        """
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.total_samples = chunk_size * num_chunks

        # Store chunks as 2D array (num_chunks, chunk_size) for easy indexing
        self.chunks = np.zeros((num_chunks, chunk_size), dtype=np.float32)
        self.chunk_index = 0

        # Pre-allocate flattened output buffer
        self.flattened_buffer = np.zeros(self.total_samples, dtype=np.float32)

        log.info(f"AudioFrameBuffer: {num_chunks} chunks × {chunk_size} samples = {self.total_samples} total samples")

    def push_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add new audio chunk to circular buffer.

        Args:
            audio_chunk: 1D array of audio samples (length must match chunk_size)
        """
        if audio_chunk.shape[0] != self.chunk_size:
            raise ValueError(f"Expected chunk size {self.chunk_size}, got {audio_chunk.shape[0]}")

        # Store chunk at current index
        self.chunks[self.chunk_index] = audio_chunk

        # Advance index
        self.chunk_index = (self.chunk_index + 1) % self.num_chunks

        # Rebuild flattened buffer in chronological order (oldest first)
        chunk_order = [(self.chunk_index + i) % self.num_chunks for i in range(self.num_chunks)]
        for i, idx in enumerate(chunk_order):
            start = i * self.chunk_size
            end = start + self.chunk_size
            self.flattened_buffer[start:end] = self.chunks[idx]

    def get_flattened_buffer(self) -> np.ndarray:
        """
        Get all audio samples in chronological order.

        Returns:
            1D array of all samples (oldest first, newest last)
        """
        return self.flattened_buffer

    def get_chunk(self, index: int) -> np.ndarray:
        """
        Get a specific chunk by its buffer index.

        Args:
            index: Index into the circular buffer (0 = oldest, num_chunks-1 = newest)

        Returns:
            1D array of audio samples for that chunk
        """
        actual_index = (self.chunk_index + index) % self.num_chunks
        return self.chunks[actual_index]

    def get_downsampled(self, target_width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get min/max envelope downsampled to target width for visualization.

        This preserves the visual shape of the waveform by keeping the min and max
        values for each window, which is what audio editors like Audacity use.

        Args:
            target_width: Number of output points (typically matches spectrogram width)

        Returns:
            Tuple of (x_values, y_min, y_max) for plotting with fill_between
        """
        audio = self.flattened_buffer
        num_samples = len(audio)

        if target_width >= num_samples:
            # No downsampling needed
            x = np.arange(num_samples)
            return x, audio, audio

        # Calculate window size
        window_size = num_samples // target_width

        # Trim to exact multiple of window_size
        trimmed_len = window_size * target_width
        trimmed_audio = audio[:trimmed_len]

        # Reshape into windows and compute min/max per window
        windowed = trimmed_audio.reshape(target_width, window_size)
        y_min = windowed.min(axis=1)
        y_max = windowed.max(axis=1)

        # X values at center of each window
        x = np.arange(target_width) * window_size + window_size // 2

        return x, y_min, y_max
