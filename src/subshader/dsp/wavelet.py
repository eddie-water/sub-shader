"""
Wavelet Module for SubShader.

This module provides Continuous Wavelet Transform (CWT) implementations for 
time-frequency analysis of audio data in real-time:
  - Creates a list of frequencies based on the chromatic scale (the typical 
    piano/musical scale we're all used to) to analyze the incoming audio data.
  - Transforms the audio data into its time-frequency representation according 
    to this list of frequencies via the CWT.
  - Each implementation is found in a subclass of the Wavelet base class,
    either a 3rd Part Library like PyWavelet, or manual implementations using
    NumPy and CuPy (GPU acceleration).
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

from abc import ABC, abstractmethod
from typing import Final, Optional, Literal

import numpy as np
from numpy.fft import fft, ifft
import cupy as cp
from cupyx.scipy import fft as cp_fft
import pywt

from subshader.utils.logging import get_logger

from ..config import WaveletConfig

# =============================================================================
# LOGGING
# =============================================================================

log = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PI: Final[float] = float(np.pi)

# =============================================================================
# WAVELET CLASSES
# =============================================================================

class Wavelet(ABC):
    def __init__(self,
                 sample_rate: np.float64,
                 input_n: int,
                 config: Optional[WaveletConfig] = None) -> None:
        """
        Wavelet base class that all other wavelet classes are derived from.
        Uses a list of frequencies that follows the chromatic scale starting at
        A0 to specify which frequencies to look for in the input audio data.

        Args:
            sample_rate: The rate the input data was sampled in Hz.
            input_n: The length of the input data (samples).
            config: Configuration object with wavelet parameters.
        """
        # Config
        if config is None:
            config = WaveletConfig()
        self.config: WaveletConfig = config

        # Sampling Parameters
        if sample_rate != self.config.typical_sampling_freq:
            log.error(f"Invalid sample rate: {sample_rate} Hz (expected {self.config.typical_sampling_freq} Hz)")
            raise ValueError(f"Sampling Rate: {sample_rate} Hz is not the typical {self.config.typical_sampling_freq} Hz. "
                             f"The CWT doesn't support non-typical sampling rates at the moment.")
        self.sample_rate: np.float64 = np.float64(sample_rate)
        self.nyquist_freq: np.float64 = self.sample_rate / 2.0
        self.sampling_period: np.float64 = 1.0 / self.sample_rate

        # Generate list of frequencies following the chromatic scale
        self.freqs: np.ndarray[np.float64] = self._generate_chromatic_scale(
            np.float64(self.config.root_note_a0_hz),
            int(self.config.num_octaves),
            int(self.config.notes_per_octave),
        ).astype(np.float64, copy=False)
        self.num_freqs: int = int(len(self.freqs))

        # Input and output dimensions
        # TODO 36 - Establish the difference between output_size and downsampled_output_size - it's fucking up the plotter
        self.input_n: int = int(input_n)
        self.input_shape: tuple[int] = (self.input_n,)
        self.output_n: int = int(self.config.target_width)
        self.output_shape: tuple[int, int] = (self.num_freqs, self.output_n)

        # Create a slice for the reliable region of the CWT output
        self.reliable_slice: slice = self._create_reliable_region_slice(
            float(self.config.reliable_mid_section_p)
        )


    def _generate_chromatic_scale(self,
                                  root_note: np.float64,
                                  num_octaves: int,
                                  notes_per_octave: int = 12) -> np.ndarray[np.float64]:
        """
        Generates a list of frequencies that follow the exponential step size of
        the chromatic scale.

        Args:
            root_note: The root note of the chromatic scale (Hz).
            num_octaves: The number of octaves to generate.
            notes_per_octave: The number of notes per octave.

        Returns:
            1D array of frequencies (Hz) in ascending order.
        """
        # Frequencies double every octave
        scale_factor = 2 ** (1 / notes_per_octave)
        i = np.arange(0, notes_per_octave * num_octaves, 1, dtype=np.float64)
        freqs = np.float64(root_note) * (scale_factor ** i)

        # Discard frequencies that are unmeasurable
        return freqs[freqs < self.nyquist_freq]

    def _create_reliable_region_slice(self, center_keep: float) -> slice:
        """
        Creates a slice that masks just for the reliable mid region of the CWT 
        output result.

        Args:
            center_keep (float): As a %, the center region to keep of the CWT 
            outputresult.

        Returns:
            slice: A mask used to slice for the reliable mid region of the CWT 
            output result.
        """
        log.info(f"Reliable Region: keeping the middle {center_keep:.1%} of the CWT result")
        result_width: int = int(self.get_output_shape()[1])
        if center_keep >= 1.0: 
            return slice(None)
        keep: int = int(round(result_width * center_keep))
        trim: int = max(0, (result_width - keep) // 2)

        # Slice that keeps the reliable middle section of the output result
        return slice(trim, trim + keep)

    def get_input_shape(self) -> tuple[int]:
        """
        Returns:
            The shape of the input data for CWT processing.
        """
        return self.input_shape

    def get_output_shape(self) -> tuple[int, int]:
        """
        Returns:
            The shape of the output data (downsampled CWT coefficients).
        """
        return self.output_shape
    
    def get_num_freqs(self) -> int:
        """
        Returns:
            The number of frequencies used in the CWT.
        """
        return self.num_freqs

    def cwt_pipeline(self, input_data: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Performs the Continuous Wavelet Transform (CWT) on input audio data,
        normalizes the results, and downsamples to produce output coefficients.

        Args:
            input_data: Raw input audio signal data; 1D array of shape (input_n,).
                        Any floating dtype accepted; internally cast to float64.

        Returns:
            2D array (num_freqs, target_width) of globally normalized magnitudes
            in the dtype specified by config.output_dtype.
        """
        # Input Validation
        if input_data.shape != self.input_shape:
            log.error(f"Input data length mismatch: {input_data.shape[0]} != {self.input_n}")
            raise ValueError(f"Input data length {input_data.shape[0]} does not match expected input data size {self.input_n}")

        # Class-Specific CWT
        cwt_coefs: np.ndarray[np.complexfloating] = self.class_specific_cwt(np.asarray(input_data, dtype=np.float64))

        # Scale-Dependent Normalization
        cwt_coefs: np.ndarray[np.complexfloating] = self.normalize_by_scale(cwt_coefs)

        # TODO 36 
        # Standardize this section: mag vs pow units, clamping (don't do this) and downsampling, and global normalization
        # Convert to magnitude or power
        mag_or_pow: np.ndarray[np.floating] = self.compute_mag_pow(cwt_coefs)

        # Avoid edge effects by extracting just reliable region
        reliable_coefs: np.ndarray[np.floating] = self.extract_reliable_region(mag_or_pow)

        # Downsample to target width
        downsampled_coefs: np.ndarray[np.floating] = self.downsample(reliable_coefs, self.output_n)

        return downsampled_coefs

    @abstractmethod
    def class_specific_cwt(self, data: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        """
        Computes the subclass-specific implementation of the CWT.

        Args:
            data: 1D float64 array of audio samples of length input_n.

        Returns:
            Complex CWT coefficients with shape (num_freqs, input_n) *before*
            reliable-region extraction and downsampling. Subclasses may choose
            to return real magnitudes if that is their native representation,
            but the parent expects complex here for consistency.
        """
        raise NotImplementedError

    def normalize_by_scale(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.complexfloating]:
        """
        Scale-Dependent Normalization to account for energy bias across scales. 
        At higher scales aka lower frequencies, the wavelet physically gets 
        wider so naturally it "collects more stuff" aka energy. To compensate 
        for that, we reduce the energy of the coefficients of a certain scale
        by its scale (s ≈ 1/f).

        Args:
            cwt_coefs: Complex CWT coefficients, shape: num_freqs x time_samples

        Returns:
            Complex CWT coefficients after scale normalization.
        """
        return cwt_coefs * np.sqrt(self.freqs[:, None])
    
    # TODO ISSUE-36: This changes the units depending on mag vs pow choice
    def compute_mag_pow(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.floating]:
        """
        Convert the CWT coefficients to magnitude or power depending on the
        config.

        Args:
            cwt_coefs: Complex CWT coefficients.

        Returns:
            Real magnitudes (|x|) or power (|x|^2) as a float array.
        """
        # TODO 36 - understand when to apply the 10 log 10. Here? or when plotting?
        if self.config.cwt_out_type == "mag":
            return np.abs(cwt_coefs)
        elif self.config.cwt_out_type == "pow":
            return np.abs(cwt_coefs) ** 2
        else:
            # Fallback: magnitude
            return np.abs(cwt_coefs)

    def extract_reliable_region(self, cwt_coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Extract the reliable center region from CWT coefficients to avoid edge artifacts.

        Args:
            cwt_coefs: Full CWT coefficients (freq_bins, time_samples).

        Returns:
            Reliable center region (freq_bins, reliable_time_samples).
        """
        reliable_region = cwt_coefs[:, self.reliable_slice]
        log.debug(
            f"Reliable Region: extracted reliable region {cwt_coefs.shape} -> {reliable_region.shape}"
        )
        return reliable_region

    def normalize_coefs(self, raw_coefs: np.ndarray[np.complexfloating | np.floating]) -> np.ndarray[np.float64]:
        """
        Normalize CWT coefficients for plotting using a fixed dB range.

        This avoids per-frame min/max scaling (which causes flicker and grain)
        by mapping magnitudes into a consistent dynamic range.

        Args:
            raw_coefs: Raw CWT coefficients (complex or real).

        Returns:
            Normalized magnitudes in [0, 1] as float32.
        """
        # Magnitude of the CWT coefficients, add epsilon to avoid log(0)
        mag = np.abs(raw_coefs) + self.config.epsilon

        # Convert to decibels
        db_vals = 20.0 * np.log10(mag)

        # Fixed display range (dB)
        db_floor = self.config.db_floor
        db_ceil = self.config.db_ceil

        # Clamp to dB range
        db_vals = np.clip(db_vals, db_floor, db_ceil)

        # Map to [0, 1]
        norm_vals = (db_vals - db_floor) / (db_ceil - db_floor)

        return norm_vals.astype(np.float64)
    
    def downsample(self,
                   coefs: np.ndarray[np.floating],
                   target_width: Optional[int] = None) -> np.ndarray[np.floating]:
        """
        Downsample CWT coefficients to produce final output data.

        This method reduces the time dimension while preserving frequency resolution
        to make the data suitable for real-time GPU rendering.

        Args:
            coefs: Input CWT coefficients (freq_bins, time_samples).
            target_width: Target output width (uses config if None).

        Returns:
            Output data suitable for visualization with shape (freq_bins, target_width).
        """
        # Use config target width if not specified
        if target_width is None:
            target_width = self.config.target_width

        freq_bins, time_samples = coefs.shape  # type: ignore[assignment]
        
        # If already at target size or smaller, return as-is
        if time_samples <= target_width:
            return coefs
        
        # Calculate downsampling factor
        downsample_factor = max(1, time_samples // int(target_width))

        # Simple downsampling strategy - take every Nth sample
        # This preserves the most recent data (right side of the buffer)
        downsampled = coefs[:, ::downsample_factor]
        
        # If still too wide, crop to target size
        if downsampled.shape[1] > target_width:
            downsampled = downsampled[:, -int(target_width):]  # Keep most recent data
        
        log.debug(
            f"Downsampled to output data: {coefs.shape} -> {downsampled.shape} (factor: {downsample_factor})"
        )
        return downsampled

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources used by the wavelet implementation.

        This method should be overridden by subclasses that allocate
        significant resources (especially GPU memory).
        """
        raise NotImplementedError

class WaveletKernel():
    def __init__(self,
                 f: np.float64,
                 input_n: int,
                 sample_rate: int,
                 num_cycles: int,
                 num_fwhm_cycles: int) -> tuple[list[np.ndarray[np.complex64]], list[np.ndarray[np.complex64]]]:

        self.center_freq: np.float64 = f

        # Wavelets in time/freq domain and their length attributes
        self.kernel_t: np.ndarray[np.complex64] = None
        self.kernel_f: np.ndarray[np.complex64] = None
        self.kernel_n: int = None
        self.half_kernel_n: int = None
        self.conv_n: Optional[int] = None

        # Wavelet duration in sec for the number of cycles at this center frequency
        wavelet_dur_s: np.float64 = num_cycles / self.center_freq

        # Number of samples in the kernel (s * samples / s)
        wavelet_n: int = int(np.round(wavelet_dur_s * sample_rate))

        # Time vector centered at t = 0
        t: np.ndarray[np.float64] = (np.arange(wavelet_n, dtype=np.float64) / sample_rate) - (wavelet_dur_s / 2)

        # Gaussian bell duration in sec for the width of the curve where the energy > half the max
        fwhm_dur_s: np.float64 = num_fwhm_cycles / self.center_freq

        # Complex Morlet Wavelet in the time domain: sinusoid * Gaussian bell curve
        sinusoid: np.ndarray[np.complex64] = np.exp(1j * 2 * PI * f * t)
        gaussian: np.ndarray[np.complex64] = np.exp(-4 * np.log(2) * (t ** 2) / fwhm_dur_s ** 2)
        cmw: np.ndarray[np.complex64] = sinusoid * gaussian

        self.kernel_t = cmw.astype(np.complex64)
        self.kern_n = int(len(self.kernel_t))
        self.conv_n = int(input_n + self.kern_n - 1)
        self.half_kern_n = int(self.kern_n // 2)
        self.slice = slice(self.half_kern_n, self.half_kern_n + input_n)

        # Store the frequency domain representation of the wavelet kernel
        self.kernel_f = fft(self.kernel_t, self.conv_n)

        # GPU-specific attributes
        self.kernel_f_gpu: Optional[cp.ndarray] = None
        self.slice_gpu: slice = self.slice 

    def upload_to_gpu(self) -> None:
        """
        Upload the wavelet kernel to the GPU.
        """
        self.kernel_f_gpu = cp.asarray(self.kernel_f, dtype=cp.complex64)

class PyWavelet(Wavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: Optional[WaveletConfig] = None) -> None:
        """
        The PyWavelet implementation of the CWT.

        Args:
            sample_rate: The rate the input data was sampled in Hz.
            input_n: The length of the input data.
            config: Configuration object with wavelet parameters.
        """
        super().__init__(sample_rate, input_n, config)

        # Wavelet info TODO ISSUE-36 why 1.5-1.0?
        self.wavelet_name: str = "cmor1.5-1.0"

        # Scale array used to specify wavelet dilation amounts during CWT
        f_norm: np.ndarray[np.float64] = (self.freqs / self.sample_rate)
        self.scales: np.ndarray[np.float64] = pywt.frequency2scale(self.wavelet_name, f_norm)

    def class_specific_cwt(self, data: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        """
        Produces the normalized CWT coefficients using PyWavelets.

        Args:
            data: 1D float64 array of audio samples.

        Returns:
            Scale-normalized complex CWT coefficients (num_freqs, input_n).
        """
        coefs_raw, _ = pywt.cwt(
            data=data,
            scales=self.scales,
            wavelet=self.wavelet_name,
            sampling_period=self.sampling_period,
        )
    
        return coefs_raw
    
    def cleanup(self) -> None:
        """PyWavelet doesn't allocate significant resources, so this is a no-op."""
        return None

class AntsWavelet(Wavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: Optional[WaveletConfig] = None) -> None:
        """
        ANTS-style CWT with true scale-dependent time support.

        Args:
            sample_rate: Input audio sample rate in Hz.
            input_n: Input data length in samples.
            num_cycles: Number of carrier cycles per wavelet.
            fwhm_cycles: Gaussian FWHM width in cycles.
            config: Configuration object with wavelet parameters.
        """
        super().__init__(sample_rate, input_n, config)

        self.wavelets: list[WaveletKernel] = []

        for f in self.freqs:
            self.wavelets.append(WaveletKernel(f, self.input_n, self.sample_rate, config.num_cycles, config.num_fwhm_cycles))

        self.num_wavelets = len(self.wavelets)

class NumPyWavelet(AntsWavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: Optional[WaveletConfig] = None) -> None:
        """NumPy-based CWT with true scale-dependent time support."""
        super().__init__(sample_rate, input_n, config)

        self.scale_bias: np.ndarray[np.float64] = np.sqrt(self.freqs).astype(np.float64)

    def class_specific_cwt(self, input_t: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        """
        Perform CWT using variable-length wavelets, CPU version.

        Args:
            input_t: 1D float64 array of time-domain audio samples.

        Returns:
            Real-valued TF matrix (num_freqs, input_n) 
        """
        output_tf: np.ndarray[np.complex64] = np.zeros((self.num_freqs, self.input_n), dtype=np.complex64)

        # Convolve input data and each wavelet kernel via frequency domain multiplication
        for i, w in enumerate(self.wavelets):
            input_f = fft(input_t, w.conv_n)
            conv = ifft(input_f * w.kernel_f)
            conv_valid = conv[w.slice]
            output_tf[i, :] = conv_valid

        return output_tf

    def cleanup(self) -> None:
        return None


class CuPyWavelet(AntsWavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: Optional[WaveletConfig] = None) -> None:
        super().__init__(sample_rate, input_n, config)

        log.info(f"CPU→GPU: Uploading {self.num_wavelets} wavelets to GPU")

        for w in self.wavelets:
            w.upload_to_gpu()

        # TODO-36 NEXT - what's the purpose of this being here?
        self.scale_bias = np.sqrt(self.freqs).astype(np.float64)
        self.scale_bias = cp.asarray(self.scale_bias, dtype=cp.float32)

        # Allocate GPU time-frequency matrix
        self.tf_gpu: cp.ndarray = cp.zeros((self.num_freqs, self.input_n), dtype=cp.complex64)

    def class_specific_cwt(self, input_t: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        """
        Perform CWT using variable-length wavelets, GPU version.

        Args:
            data: 1D float64 array of audio samples.

        Returns:
            Real-valued TF matrix (num_freqs, input_n) containing power (|x|^2),
            transferred back to NumPy on return.
        """
        input_t_cp = cp.asarray(input_t, dtype=cp.complex64)

        for i, w in enumerate(self.wavelets):
            input_f = cp_fft.fftn(input_t_cp, w.conv_n)
            conv = cp_fft.ifftn(input_f * w.kernel_f_gpu)
            conv_valid = conv[w.slice_gpu]
            self.tf_gpu[i, :] = conv_valid

        return cp.asnumpy(self.tf_gpu)

    # TODO NOW - remove this?
    # TODO 36 why is this done in two different places?
    def normalize_globally(self, raw_coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Apply global normalization for CuPy-based wavelets.

        Args:
            raw_coefs: Raw CWT coefficients (already converted from GPU, NumPy array).

        Returns:
            Globally normalized magnitudes in [0, 1].
        """
        if self.global_normalizer is None:
            return np.asarray(raw_coefs, dtype=self.config.output_dtype)
        return super().normalize_globally(raw_coefs)

    def cleanup(self) -> None:
        try:
            if hasattr(self, 'tf_gpu') and self.tf_gpu is not None:
                del self.tf_gpu
                self.tf_gpu = None  # type: ignore[assignment]
            if hasattr(self, 'wavelet_kernels_f') and self.wavelet_kernels_f is not None:
                del self.wavelet_kernels_f
                self.wavelet_kernels_f = []  # type: ignore[assignment]
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:  # pragma: no cover - defensive cleanup
            print(f"Warning: Error during GPU cleanup: {e}")
    
class CuWavelet(CuPyWavelet):
    """Alias of CuPyWavelet with a shorter name."""
    pass