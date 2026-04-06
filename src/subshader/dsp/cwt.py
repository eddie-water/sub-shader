"""
CWT Module for SubShader.

Continuous Wavelet Transform backends using the ANTS (Analyzing Neural Time
Series) approach: variable-length complex Morlet wavelets convolved with the
audio signal in the frequency domain.

Class hierarchy:
    DSP -> CWT -> CpuCWT   (NumPy FFT convolution, CPU)
               -> GpuCWT   (CuPy FFT convolution, GPU)
"""

from __future__ import annotations

from typing import Final, Optional

try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False

import numpy as np
from numpy.fft import fft, ifft

from subshader.config import CWTConfig
from subshader.dsp.dsp import DSP
from subshader.dsp.wavelet_kernel import WaveletKernel
from subshader.utils.logging import get_logger
from subshader.utils.timing import timed

log = get_logger(__name__)

PI: Final[float] = float(np.pi)


class CWT(DSP):
    """
    CWT base class implementing the ANTS wavelet convolution algorithm.

    On init, constructs a bank of complex Morlet wavelet kernels covering the
    chromatic scale and determines the reliable output region. Subclasses
    implement transform() with their specific FFT convolution (CPU or GPU).
    """

    def __init__(self, config: CWTConfig) -> None:
        super().__init__(config)

        self.sample_rate: np.float64 = np.float64(config.sample_rate)
        self.nyquist_freq: np.float64 = self.sample_rate / 2.0
        self.sampling_period: np.float64 = 1.0 / self.sample_rate
        self.overlap_factor: float = float(config.overlap_factor)

        # Input / output dimensions
        self.input_n: int = int(config.chunk_size)
        self.input_shape: tuple = (self.input_n,)
        self.output_n: int = config.target_width

        # Build chromatic frequency scale
        self.freqs: np.ndarray = self._generate_chromatic_scale(
            config.root_note_hz,
            config.num_octaves,
            config.notes_per_octave,
        ).astype(np.float64, copy=False)

        self.num_freqs: int = len(self.freqs)
        self.output_shape: tuple = (self.num_freqs, self.output_n)

        # Build wavelet kernel bank
        self.wavelets: list[WaveletKernel] = [
            WaveletKernel(
                f=f,
                sample_rate=self.sample_rate,
                num_cycles=config.num_cycles,
                num_fwhm_cycles=config.num_fwhm_cycles,
                input_n=self.input_n,
            )
            for f in self.freqs
        ]

        self.num_wavelets: int = len(self.wavelets)
        self.max_conv_n: int = max(w.get_conv_n() for w in self.wavelets)

        # Pre-compute frequency-domain kernel bank (zero-padded to max_conv_n)
        self.kernel_f_bank: np.ndarray = np.zeros(
            (self.num_wavelets, self.max_conv_n), dtype=np.complex64
        )
        for i, w in enumerate(self.wavelets):
            self.kernel_f_bank[i] = fft(w.kernel_t, self.max_conv_n)

        # Reliable center slice (discards edge artifacts from widest wavelet)
        self.reliable_slice: slice = self._create_reliable_slice(self.wavelets)

    # -------------------------------------------------------------------------
    # DSP pipeline stages
    # -------------------------------------------------------------------------

    def pre(self, chunk: np.ndarray) -> np.ndarray:
        """Validate input shape and cast to float64.

        Args:
            chunk: Raw 1D audio samples from AudioStream.

        Returns:
            float64 array of length input_n.

        Raises:
            ValueError: If chunk length does not match config.chunk_size.
        """
        if chunk.shape != self.input_shape:
            log.error(f"Input shape mismatch: {chunk.shape} != {self.input_shape}")
            raise ValueError(
                f"Input data length {chunk.shape[0]} does not match "
                f"expected input size {self.input_n}"
            )
        return np.asarray(chunk, dtype=np.float64)

    def post(self, raw: np.ndarray) -> np.ndarray:
        """Post-process raw complex CWT coefficients into a visualization-ready array.

        Steps:
            1. Normalize by scale (no-op — L1 kernel normalization handles energy bias)
            2. Convert to magnitude
            3. Discard edge-artifact region
            4. Extract non-overlapping hop center
            5. Downsample to target_width

        Args:
            raw: Complex coefficients from transform(), shape (num_freqs, input_n).

        Returns:
            Float32 array of shape (num_freqs, target_width).
        """
        # Energy bias correction is handled at WaveletKernel construction (L1 normalization);
        # normalize_by_scale is a no-op kept for API symmetry.
        normed = self._normalize_by_scale(raw)
        mag = self._compute_mag(normed)
        reliable = self.discard_unreliable_coefs(mag)
        hop_center = self.extract_hop_center(reliable)
        return self.downsample(hop_center, self.output_n)

    # -------------------------------------------------------------------------
    # Abstract: implemented by CpuCWT / GpuCWT
    # -------------------------------------------------------------------------

    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement transform()")

    def get_output_shape(self) -> tuple:
        """Return the shape of the processed output."""
        return self.output_shape

    # -------------------------------------------------------------------------
    # Private helpers — chromatic scale and reliable-region construction
    # -------------------------------------------------------------------------

    def _generate_chromatic_scale(
        self,
        root_note: float,
        num_octaves: int,
        notes_per_octave: int = 12,
    ) -> np.ndarray:
        """Generate an exponentially-spaced frequency array following the chromatic scale.

        Args:
            root_note: Root frequency in Hz (e.g. A0 = 27.5 Hz).
            num_octaves: Number of octaves to span.
            notes_per_octave: Steps per octave (12 = semitone resolution).

        Returns:
            1D float64 array of frequencies below Nyquist, in ascending order.
        """
        scale_factor = 2 ** (1 / notes_per_octave)
        i = np.arange(0, notes_per_octave * num_octaves, dtype=np.float64)
        freqs = np.float64(root_note) * (scale_factor ** i)
        return freqs[freqs < self.nyquist_freq]

    def _create_reliable_slice(self, wavelets: list[WaveletKernel]) -> slice:
        """Compute the uniform center-keep slice that avoids edge artifacts.

        Uses the wavelet with the widest time support to set the keep width,
        rounded down to the nearest power of two for clean downstream math.

        Returns:
            slice spanning the reliable center region of the CWT output.
        """
        keep = 0
        for w in wavelets:
            if w.time_support_n > keep:
                log.info(
                    f"Determining reliable region for Wavelet {w.freq} Hz "
                    f"with time support: {w.time_support_n} samples"
                )
                keep = 2 ** int(np.floor(np.log2(w.time_support_n)))
                log.info(f"Keeping to the nearest power of 2: {keep}")

        slice_start = (self.input_n - keep) // 2
        slice_end = slice_start + keep
        return slice(slice_start, slice_end)

    # -------------------------------------------------------------------------
    # Post-processing steps (used in post())
    # -------------------------------------------------------------------------

    def _normalize_by_scale(self, cwt_coefs: np.ndarray) -> np.ndarray:
        """No-op: energy bias corrected at kernel construction via L1 normalization."""
        return cwt_coefs

    @timed
    def _compute_mag(self, cwt_coefs: np.ndarray) -> np.ndarray:
        """Convert complex CWT coefficients to real magnitudes."""
        return np.abs(cwt_coefs)

    @timed
    def discard_unreliable_coefs(self, coefs: np.ndarray) -> np.ndarray:
        """Slice to the reliable center region to remove edge artifacts.

        The reliable slice is determined by the widest wavelet's time support
        and is uniform across all frequency rows (rectangular output).

        Args:
            coefs: Magnitude coefficients, shape (num_freqs, input_n).

        Returns:
            Center-sliced coefficients, shape (num_freqs, reliable_width).
        """
        return coefs[:, self.reliable_slice]

    @timed
    def extract_hop_center(self, reliable_coefs: np.ndarray) -> np.ndarray:
        """Extract the non-overlapping trailing portion of the reliable region.

        When overlap_factor > 0, consecutive chunks share samples. The trailing
        hop_size columns of the reliable region represent new audio content not
        present in the previous frame, ensuring frames tile without redundancy.

        Args:
            reliable_coefs: Coefficients after discard_unreliable_coefs,
                            shape (num_freqs, reliable_width).

        Returns:
            Trailing slice of shape (num_freqs, hop_size) or input unchanged
            if overlap_factor is 0.
        """
        if self.overlap_factor <= 0.0:
            return reliable_coefs

        reliable_width = reliable_coefs.shape[1]
        hop_size = max(1, int(self.input_n * (1 - self.overlap_factor)))
        new_width = min(hop_size, reliable_width)
        return reliable_coefs[:, -new_width:]

    @timed
    def downsample(
        self,
        coefs: np.ndarray,
        target_width: Optional[int] = None,
    ) -> np.ndarray:
        """Downsample the time dimension to target_width by uniform index selection.

        Args:
            coefs: Input coefficients, shape (freq_bins, num_samples).
            target_width: Target number of time bins.

        Returns:
            Downsampled coefficients, shape (freq_bins, target_width).

        Raises:
            ValueError: If target_width is out of range.
        """
        if target_width is None:
            target_width = self.output_n

        _, num_samples = coefs.shape

        if target_width <= 0 or target_width > num_samples:
            raise ValueError(
                f"Invalid target width: {target_width} "
                f"(must be between 1 and {num_samples})"
            )

        hop = num_samples / target_width
        indices = np.floor(np.arange(target_width) * hop).astype(int)
        indices = np.clip(indices, 0, num_samples - 1)
        downsampled = coefs[:, indices]

        log.debug(f"Downsampled: {coefs.shape} -> {downsampled.shape}")
        return downsampled

    def cleanup(self) -> None:
        """Release resources. Subclasses override for GPU memory cleanup."""
        pass


class CpuCWT(CWT):
    """NumPy-based CWT using FFT convolution on CPU.

    Computes the full-bank FFT convolution using numpy.fft: input is
    transformed once, multiplied against all kernel rows, then iFFT'd.
    """

    def __init__(self, config: CWTConfig) -> None:
        super().__init__(config)

    @timed
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Perform CWT via NumPy FFT convolution (CPU).

        Args:
            data: 1D float64 audio samples of length input_n.

        Returns:
            Complex TF matrix (num_freqs, input_n).
        """
        input_f = fft(data, self.max_conv_n)
        conv_f = input_f * self.kernel_f_bank
        conv_tf = ifft(conv_f, axis=1)
        return conv_tf[:, : self.input_n]

    def cleanup(self) -> None:
        return None


class GpuCWT(CWT):
    """CuPy-based CWT using FFT convolution on GPU.

    Uploads the kernel bank to GPU at init. Each transform() call uploads
    the input, performs the full convolution on-device, and downloads only
    the trimmed result (num_freqs × input_n) to avoid a large intermediate
    transfer of shape (num_freqs × max_conv_n).
    """

    def __init__(self, config: CWTConfig) -> None:
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not available — cannot create GpuCWT. "
                "Use CpuCWT for CPU-only execution."
            )
        super().__init__(config)

        log.info(f"CPU→GPU: Uploading {self.num_wavelets} wavelet kernels to GPU")
        self.kernel_f_bank_gpu = cp.asarray(
            self.kernel_f_bank, dtype=cp.complex64, order="C"
        )

    @timed
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Perform CWT via CuPy FFT convolution (GPU).

        Args:
            data: 1D float64 audio samples of length input_n (CPU array).

        Returns:
            Complex TF matrix (num_freqs, input_n) as NumPy array (CPU).
        """
        log.info(f"CPU→GPU: Uploading input data of size {data.shape[0]}")
        input_f_gpu = cp.asarray(
            fft(data, self.max_conv_n), dtype=cp.complex64, order="C"
        )

        conv_f_gpu = input_f_gpu * self.kernel_f_bank_gpu
        conv_tf_gpu = cp_fft.ifft(conv_f_gpu, axis=1)
        conv_tf_trimmed = conv_tf_gpu[:, : self.input_n]

        log.info(f"GPU→CPU: Transferring trimmed result {conv_tf_trimmed.shape}")
        return cp.asnumpy(conv_tf_trimmed)

    def cleanup(self) -> None:
        """Free GPU memory pool allocations."""
        try:
            if hasattr(self, "kernel_f_bank_gpu") and self.kernel_f_bank_gpu is not None:
                del self.kernel_f_bank_gpu
            if _CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:  # pragma: no cover - defensive cleanup
            log.warning(f"Error during GPU cleanup: {e}")
