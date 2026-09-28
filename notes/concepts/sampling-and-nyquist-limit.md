# Sampling and Nyquist Limit

Sampling rate sets how far apart in time consecutive samples are, and the Nyquist limit says at least two samples per cycle are needed to detect a frequency — so the highest detectable frequency is half the sample rate. Window duration controls the lowest measurable frequency and how finely nearby frequencies can be told apart: longer windows resolve lower frequencies and separate closer ones.

**Key numbers**

- Audio sample rate: 44.1 kHz → Nyquist ceiling at **22.05 kHz** (`nyquist_freq = sample_rate / 2` in [`config.py`](../../src/subshader/config.py))
- Analysis chunk: 16,384 samples = **371.5 ms** of signal — long enough to contain the full 6-cycle wavelet support of the lowest note, A0 at 27.5 Hz (218 ms)
- The [chromatic scale](chromatic-frequency-scale.md) is explicitly clipped below Nyquist: of 120 candidate semitones (10 octaves × 12), the top 4 exceed 22.05 kHz, leaving **116**

## In SubShader

Covered in [DSP §3.1](../../src/subshader/dsp/DSP.md#31-sampling-rate-and-duration). Enforced in two places in [`cwt.py`](../../src/subshader/dsp/cwt.py): `_generate_chromatic_scale()` drops frequencies at or above Nyquist, and the `chunk_size` default in [`config.py`](../../src/subshader/config.py) is sized so even the widest wavelet fits inside one window.

---

**Related:** [Basis Functions](basis-functions.md) · [Time-Frequency Resolution Tradeoff](time-frequency-resolution-tradeoff.md) · [Chromatic Frequency Scale](chromatic-frequency-scale.md) · [MAP](../MAP.md)
