TODO-37 Explain the wavlet design with nice figures

# Wavelet Transform Design Intuitions

This module implements a Continuous Wavelet Transform (CWT) framework optimized for audio chunks of a constant length, using variable-length wavelet kernels designed for perceptual accuracy and time-frequency localization.

## Goal

Analyze audio for musical content and produce an accurate time-frequency representation.

---

## Design Assumptions

- **Sample Rate:** 44.1 kHz - Most audio is recorded at this sample rate
- **Frequency Range:** Chromatic Scale - The frequencies we are interested in analyzing belong to the chromatic scale. This is the typical music scale used in mainstream music ()
- **Input Chunk Size:** 4096 samples (~93 ms)
- **Wavelet Cycles:** 6 cycles per wavelet to maintain oscillatory resolution

---

## Wavelet Duration and Time Support

Each wavelet kernel is constructed in the time domain to span a minimum of 6 oscillatory cycles. The time-domain support (duration) for a given frequency `f` is:

```
duration_s = 6 / f
```

This ensures adequate frequency resolution while maintaining localizability. The number of samples per wavelet is:

```
wavelet_n = duration_s * sample_rate
```

This is stored as `wavelet_n` in each `WaveletKernel` instance and used to determine edge effects.

---

## Output Shape of the CWT

- The CWT result always has shape `(num_scales, input_n)` where `input_n = 4096`.
- This holds regardless of wavelet length — but only the center region of each row is reliable.

---

## Cone of Influence (COI)

The cone of influence marks the edge regions of the CWT result that are affected by boundary distortion due to finite-length wavelets.

For each wavelet:

```
margin_samples = wavelet_n // 2
```

Only samples in the range `[margin : -margin]` for that scale's CWT row are reliable. These margins are applied **per-scale** to build a 2D COI mask.

This scale-aware masking allows precise slicing of valid data for analysis and visualization.

---

## Practical Tradeoffs and Filtering

In practice:

- Low-frequency wavelets (e.g., 27.5 Hz) may have durations longer than the input chunk
- These scales result in very narrow or nonexistent valid regions
- Such wavelets are optionally skipped from computation if:

```
wavelet_n > input_n
```

This avoids wasting compute on rows that would be fully masked out.

---

## Recommendations for Real-Time Use

To balance accuracy and performance:

- Consider reducing the number of cycles per wavelet (e.g., 3–4 instead of 6)
- Increase input chunk size (e.g., 8192 samples) to support low frequencies
- Set a low-frequency floor (e.g., 40–60 Hz) to avoid unreliable rows

---

## Summary

- All wavelets produce CWT rows of the same length (input size)
- The valid (reliable) portion of each row is determined by the wavelet's time-domain support
- The cone of influence is implemented as a 2D scale-aware mask
- Wavelets that are too wide for the input are filtered out to avoid redundant computation

