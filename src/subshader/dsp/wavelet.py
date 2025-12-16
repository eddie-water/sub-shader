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

import cupy as cp
from cupyx.scipy import fft as cp_fft
import numpy as np
from numpy.fft import fft, ifft
import pywt

from subshader.utils.logging import get_logger
from subshader.dsp.wavelet_kernel import WaveletKernel

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
                 config: WaveletConfig) -> None:
        """
        Wavelet base class that all other wavelet classes are derived from.
        Generates a list of frequencies that follows the chromatic scale that
        specify which frequencies to look for in the input audio data.

        Args:
            sample_rate: The rate the input data was sampled in Hz.
            input_n: The length of the input data (samples).
            config: Configuration object with wavelet parameters.
        """
        self.config: WaveletConfig = config

        if sample_rate != self.config.typical_sampling_freq:
            log.error(f"Invalid sample rate: {sample_rate} Hz (expected {self.config.typical_sampling_freq} Hz)")
            raise ValueError(f"Sampling Rate: {sample_rate} Hz is not the typical {self.config.typical_sampling_freq} Hz."
                             f"The CWT doesn't support non-typical sampling rates at the moment.")

        # Store params
        self.sample_rate: np.float64 = np.float64(sample_rate)
        self.nyquist_freq: np.float64 = self.sample_rate / 2.0
        self.sampling_period: np.float64 = 1.0 / self.sample_rate

        # Generate list of frequencies following the chromatic scale
        self.freqs: np.ndarray[np.float64] = self._generate_chromatic_scale(self.config.root_note_a0_hz,
                                                                            self.config.num_octaves,
                                                                            self.config.notes_per_octave).astype(np.float64, copy=False)

        self.num_freqs: int = int(len(self.freqs))

        # Input and output dimensions
        self.input_n: int = int(input_n)
        self.input_shape: tuple[int] = (self.input_n,)
        self.output_n: int = self.config.target_width
        self.output_shape: tuple[int, int] = (self.num_freqs, self.output_n)

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

    def get_input_shape(self) -> tuple[int]:
        """
        Returns:
            The shape of the input data for CWT processing.
        """
        return self.input_shape

    def get_output_shape(self) -> tuple[int]:
        """
        Determines the shape of the output data for the CWT.
        """
        return self.output_shape

    def cwt(self, input_data: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
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
        if input_data.shape != self.input_shape:
            log.error(f"Input data length mismatch: {input_data.shape[0]} != {self.input_n}")
            raise ValueError(f"Input data length {input_data.shape[0]} does not match expected input data size {self.input_n}")

        # Class-Specific CWT
        cwt_coefs = self.class_specific_cwt(np.asarray(input_data, dtype=np.float64))

        # Scale-Dependent Normalization
        cwt_coefs = self.normalize_by_scale(cwt_coefs)

        # Convert to magnitude
        mag_coefs = self.compute_mag(cwt_coefs)

        # Keep only the reliable region to avoid edge effects
        reliable_coefs = self.discard_unreliable_coefs(mag_coefs)

        # Downsample to target width
        downsampled_coefs = self.downsample(reliable_coefs, self.output_n)

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

    @abstractmethod
    def normalize_by_scale(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.complexfloating]:
        """
        Scale-Dependent Normalization to account for the energy bias introduced
        by scaling each wavelet. At higher scales (lower frequencies), the 
        wavelet physically gets wider so naturally it collects more energy. To 
        compensate for that, we reduce the energy of the cwt's result by square 
        root of the scale where s ≈ 1/f -> 1/sqrt(s) ≈ sqrt(f) 

        Args:
            cwt_coefs: Complex CWT coefficients.

        Returns:
            Scale-normalized complex CWT coefficients.
        """
        raise NotImplementedError

    def compute_mag(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.floating]:
        """
        Convert the raw CWT coefficients to magnitude 

        Args:
            cwt_coefs: Complex CWT coefficients.

        Returns:
            Real magnitudes (|x|) as a float array.
        """
        return np.abs(cwt_coefs)

    @abstractmethod
    def discard_unreliable_coefs(self, coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Discards the unreliable coefficients outside the reliable region of the
        CWT output. 

        Args:
            coefs: Result time-frequency coefficients from the CWT.

        Returns:
            The reliable time-frequency coefficients.
        """
        raise NotImplementedError
    
    def downsample(self,
                   coefs: np.ndarray[np.floating],
                   target_width: Optional[int] = None) -> np.ndarray[np.floating]:
        """
        Downsample CWT coefficients to produce final output data.

        This method reduces the time dimension 

        Args:
            coefs: Input CWT coefficients (freq_bins, num_samples).
            target_width: Target output width.

        Returns:
            Output data suitable for visualization with shape (freq_bins, target_width).
        """
        if target_width is None:
            target_width = self.output_n

        _, num_samples = coefs.shape

        # Input Validation
        if target_width <= 0 or target_width > num_samples:
            raise ValueError(f"Invalid target width: {target_width} (must be between 0 and {num_samples})")

        # Fractional hop size
        hop = num_samples / target_width

        # Determine the indices to keep using hop size
        indices = np.floor(np.arange(target_width) * hop).astype(int)

        indices = np.clip(indices, 0, num_samples - 1)
        downsampled = coefs[:, indices]

        log.debug(f"Downsampled to output data: {coefs.shape} -> {downsampled.shape}")

        return downsampled

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources used by the wavelet implementation.

        This method should be overridden by subclasses that allocate
        significant resources (especially GPU memory).
        """
        raise NotImplementedError

class PyWavelet(Wavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: WaveletConfig) -> None:
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
    
    def normalize_by_scale(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.complexfloating]:
        """
        Scale-Dependent Normalization to account for the energy bias introduced
        by scaling each wavelet.

        Args:
            cwt_coefs: Complex CWT coefficients.

        Returns:
            Scale-normalized complex CWT coefficients.
        """
        return cwt_coefs

    def discard_unreliable_coefs(self, coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        # stub
        keep = 8192
        slice_start: int = (self.input_n - keep) // 2
        slice_end: int = slice_start + keep
        return coefs[:, slice_start:slice_end]
    
    def cleanup(self) -> None:
        """PyWavelet doesn't allocate significant resources, so this is a no-op."""
        return None

class AntsWavelet(Wavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: WaveletConfig = None) -> None:
        """
        CWT implementation from Analyzing Neural Time Series (ANTS) by Mike X 
        Cohen translated from Matlab. According to the list of frequencies, we
        generate a bank of wavelet kernels on init. Then determine the output's
        reliable region based on the wavelet with the widest time support. All
        of this in preparation for the CWT call, which performs the convolution
        between the audio data and the bank of wavelet kernels.

        Args:
            sample_rate: Input audio sample rate in Hz.
            input_n: Input data length in samples.
            num_cycles: Number of carrier cycles per wavelet.
            fwhm_cycles: Gaussian FWHM width in cycles.
            config: Configuration object with wavelet parameters.
        """
        super().__init__(sample_rate, input_n, config)

        # Generate the each wavelet
        self.wavelets: list[WaveletKernel] = [WaveletKernel(f=f, 
                                                            sample_rate=self.sample_rate, 
                                                            num_cycles=self.config.num_cycles, 
                                                            num_fwhm_cycles=self.config.num_fwhm_cycles, 
                                                            input_n=self.input_n) for f in self.freqs]

        self.num_wavelets: int = len(self.wavelets)

        # Store each kernel in the bank where each kernel is zero-padded to the longest kernel's conv length
        self.max_conv_n: int = max(w.get_conv_n() for w in self.wavelets)

        self.kernel_f_bank: np.ndarray[np.complex64] = np.zeros((self.num_wavelets, self.max_conv_n), dtype=np.complex64)

        for i, w in enumerate(self.wavelets):
            self.kernel_f_bank[i, :w.get_conv_n()] = w.kernel_f

        # Assess the reliable regions of the CWT output
        # self.coi_mask: np.ndarray[bool] = self._create_coi_mask(self.wavelets)
        self.reliable_slice: slice = self._create_reliable_slice(self.wavelets)

    def _create_coi_mask(self, wavelets: list[WaveletKernel]) -> np.ndarray[bool]:
        """
        Creates a mask marking the center (reliable) region of the CWT result 
        for each wavelet. For each frequency row, only the central segment—the 
        length of the wavelet's time support (i.e., the number of samples 
        spanned by its time support)—is kept as reliable; edges are masked out, 
        as wavelets do not provide accurate energy information there.
        """
        mask = np.zeros((len(wavelets), self.input_n), dtype=bool)

        for i, w in enumerate(wavelets):
            margin: int = w.time_support_n // 2
            start: int =  margin
            stop: int = w.input_n - margin
            log.info(f"Wavelet Mask {i} margin: {margin}, start: {start}, stop: {stop}")
            mask[i, start:stop] = True

        return mask

    def _create_reliable_slice(self, wavelets: list[WaveletKernel]) -> slice:
        """
        Uses the wavelet with the widest time support to create a center keep
        slice for the reliable region of the CWT output.

        Returns:
            slice: A slice used to mask the reliable center region of the CWT 
            result.
        """

        keep: int = 0

        for w in wavelets:
            if w.time_support_n > keep:
                log.info(f"Determining reliable region for Wavelet {w.freq} Hz with time support: {w.time_support_n} samples")
                keep = 2 ** int(np.floor(np.log2(w.time_support_n)))
                log.info(f"Keeping to the nearest power of 2: {keep}")

        # Slice indices for the reliable center region of the CWT output 
        slice_start: int = (self.input_n - keep) // 2
        slice_end: int = slice_start + keep

        return slice(slice_start, slice_end)

    def normalize_by_scale(self, cwt_coefs: np.ndarray[np.complexfloating]) -> np.ndarray[np.complexfloating]:
        """
        Scale-Dependent Normalization to account for the energy bias introduced
        by scaling each wavelet. At higher scales (lower frequencies), the 
        wavelet physically gets wider so naturally it collects more energy. To 
        compensate for that, we reduce the energy of the cwt's result by square 
        root of the scale where s ≈ 1/f -> 1/sqrt(s) ≈ sqrt(f) 
        
        TODO-37 explain the square root is because power is mag^2

        Args:
            cwt_coefs: Complex CWT coefficients.

        Returns:
            Scale-normalized complex CWT coefficients.
        """
        return cwt_coefs * np.sqrt(self.freqs[:, None])

    def apply_coi_mask(self, coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Apply the Cone of Influence mask for reliable region of the CWT results
        to avoid edge artifacts from the variable length wavelets.

        Args:
            coefs: CWT output coefficients.

        Returns:
            Masked CWT output coefficients.
        """
        return coefs * self.coi_mask

    def slice_for_reliable_region(self, coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Slices the CWT output for the reliable region of the CWT output.

        Args:
            coefs: CWT output coefficients.

        Returns:
            Sliced CWT output coefficients.
        """
        return coefs[:, self.reliable_slice]

    def discard_unreliable_coefs(self, coefs: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Discards the unreliable coefficients outside the reliable region of the
        CWT output.

        Edge Effects:
            After performing the CWT, portions of the output near the signal 
            boundaries are unreliable due to edge effects in the convolution. At
            the start and end of the input, the wavelet kernel extends beyond 
            the signal into regions of zero energy. During convolution, the 
            kernels accumulate 'artificial' zero-energy contributions, 
            influencing and effectively weakening the CWT results each edge. 
            
            There are also edge effects where the samples abruptly change from a
            zero to non-zero value at the starting edge, or a non-zero to zero 
            value at the ending edge. These abrupt changes in value inflate the 
            CWT result with 'artificial' high-energy because it thinks it's 
            measuring a high-frequency component of the signal.

            Because each wavelet has a different time support (width), this edge 
            effect varies by scale - wider, high-scale/low-frequency wavelets are 
            more susceptible to this edge effect than narrow, low-scale/high-
            frequency ones. This is because the wider the wavelet, the left wing 
            of the wavelet extends itself further before the starting edge of 
            the signal, and the right wing extends itself further after the 
            ending edge.

        Mitigating Edge Effects:
            To mitigate artificial contributions for the CWT results, we can 
            determine which parts of each convolution are reliable based on its 
            wavelet's time support, discarding the unreliable edge regions. 
            Aggregating these reliable regions across the scales forms the Cone 
            of Influence (COI) - a mask that preserves CWT results uninfluenced 
            by edge effects.

            Unfortunately, using the COI to discard unreliable results produces 
            another, somewhat undesirable effect: it produces an irregularly
            shaped result. 
            
            TODO-37 Insert COI Mask Figure Here
            
            Low scales wavelets have narrower time supports, 
            less edge effects and less unreliable results. High scale wavlets 
            have wider time supports, more edge effects, and more unreliable
            results. The amount of reliable results to preserve varies by 
            scale, and discarding the unreliable results by scale-dependent 
            time-support produces a ragged, pinched, cone-shaped output. Each 
            row retains a different number of samples in the time dimension, and
            this irregular shape makes it awkward to visualize and unsuitable 
            for rectangular plotting.
            
            To maintain a consistent rectangular output, we instead trim the CWT 
            using the widest wavelet's time support, keeping a uniform central 
            region to keep across all scales. 
            
            TODO-37 Insert Center Keep Slice Figure Here
            
            Although this approach discards some valid data from high-frequency 
            wavelets, it still removes energy bias from the edges and produces
            a clean, usable output shape.
            
        Args:
            coefs: CWT output coefficients.

        Returns:
            Reliable CWT results.
        """
        # return self.apply_coi_mask(coefs)
        return self.slice_for_reliable_region(coefs)

    @abstractmethod
    def class_specific_cwt(self, data: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        pass

class NumPyWavelet(AntsWavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: WaveletConfig = None) -> None:
        """NumPy-based CWT with true scale-dependent time support."""
        super().__init__(sample_rate, input_n, config)

    def class_specific_cwt(self, input_t: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        """
        Perform CWT using variable-length wavelets, CPU version.

        Args:
            input_t: 1D float64 array of time-domain audio samples.

        Returns:
            Real-valued TF matrix (num_freqs, input_n) 
        """
        # Transform input data to frequency domain
        input_f = fft(input_t, self.max_conv_n)

        # Perform convolution per row as frequency domain multiplication
        conv_f = input_f * self.kernel_f_bank

        conv_tf = ifft(conv_f, axis=1)

        return conv_tf[:, :self.input_n]

    def cleanup(self) -> None:
        return None

class CuPyWavelet(AntsWavelet):
    def __init__(self,
                 sample_rate: int,
                 input_n: int,
                 config: WaveletConfig = None) -> None:
        super().__init__(sample_rate, input_n, config)

        log.info(f"CPU→GPU: Uploading {self.num_wavelets} wavelets to GPU")

        # Upload wavelets to GPU
        self.kernel_f_bank_gpu = cp.asarray(
            self.kernel_f_bank, 
            dtype=cp.complex64, 
            order='C')

    def class_specific_cwt(self, input_t_cpu: np.ndarray[np.float64]) -> np.ndarray[np.complexfloating]:
        """
        Perform CWT using variable-length wavelets, GPU version.

        Args:
            data: 1D float64 array of audio samples.

        Returns:
            Real-valued TF matrix (num_freqs, input_n) containing power (|x|^2),
            transferred back to NumPy on return.
        """
        log.info(f"CPU→GPU: Uploading input data of size {input_t_cpu.shape[0]} to GPU")
        
        self.input_f_gpu = cp.asarray(fft(input_t_cpu, self.max_conv_n), 
                                      dtype=cp.complex64,
                                      order='C')

        # Perform convolution via broadcasting frequency domain multiplication per row 
        conv_f_gpu = self.input_f_gpu * self.kernel_f_bank_gpu

        log.info(f"GPU→CPU: Convolution result of size {conv_f_gpu.shape} to CPU")
        conv_f_cpu = cp.asnumpy(conv_f_gpu)

        conv_tf_cpu = ifft(conv_f_cpu, axis=1)

        return conv_tf_cpu[:, :self.input_n]

    def cleanup(self) -> None:
        try:
            if hasattr(self, 'tf_gpu') and self.output_tf_cpu is not None:
                del self.output_tf_cpu
                self.output_tf_cpu = None  # type: ignore[assignment]
            if hasattr(self, 'wavelet_kernels_f') and self.wavelet_kernels_f is not None:
                del self.wavelet_kernels_f
                self.wavelet_kernels_f = []  # type: ignore[assignment]
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:  # pragma: no cover - defensive cleanup
            print(f"Warning: Error during GPU cleanup: {e}")
    

class NpWavelet(NumPyWavelet):
    """Alias for NumPyWavelet."""
    pass

class CuWavelet(CuPyWavelet):
    """Alias forCuPyWavelet"""
    pass