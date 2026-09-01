<!-- PREVIEW COPY — README.md + locked Deadline/SEAM-1 prose from the 2026-08-31 authoring session. Not the real README; delete when merged back. -->
# SubShader

SubShader is a **real-time audio visualizer** written in Python. It uses modern techniques in digital signal processing and GPU programming to accurately **visualize what an audio signal sounds like**. But to describe it as just a visualizer would be a massive understatement - it implements the foundation of a GPU-accelerated pipeline custom-built for real-time signal processing and feature extraction.

The signal processing is **[Wavelet](https://youtu.be/jnxqHcObNK4?si=x98elLTbz6QLe03g&t=1996)**-based - a modern adaptation of the traditional **[Fourier](https://youtu.be/spUNpyF58BY?si=jXTsOaIHUwB8meoc)**-based methods - converting audio into a **time-frequency** representation in real time. The advantages and justification of using it for real-world signal processing are discussed in this project.

> ℹ️ For in-depth implementation decisions and design intuitions → **[DSP README](src/subshader/dsp/DSP.md)**

<!-- HERO DEMO CLIP — GitHub attachment upload of assets/video/subshader_demo.mp4.
     Re-rendering the clip means re-uploading (drag into the github.com editor) and
     swapping this URL. -->
https://github.com/user-attachments/assets/16e7cd7a-0a5c-4d4e-8199-85d9140d2d12

Demonstrating my technical skills in real-time signal analysis and GPU acceleration, I'm using this project to branch into digital signal processing (DSP), machine learning (ML), data science, and computer engineering. **It took a lot of effort and care to make this, so thank you for taking the time to read!**

## Problem Overview

Our ears can easily distinguish all the different kinds of noise we hear - the loud vs the quiet, the high vs the low, the sudden vs the gradual - but designing something that can analyze audio to this level of detail is not as trivial as it may seem. We need to know more than just **what frequencies** are present. We need to know **when in time** they start, how long they last, and when they stop, all relative to how loud they actually are. 

The difficulty of this stems from signal details that **coexist at vastly different scales** - high-frequency activity can be extremely brief and abrupt, demanding **fine resolution in time**, while low-frequency content develops slowly and gradually, demanding **fine resolution in frequency** instead. Simultaneously capturing both ends of the spectrum in a fully-detailed representation is the core problem this project confronts.

### Audio Example

The audio here is a "non-stationary" signal - an exaggeration of this resolution problem specifically. It **gently** sweeps a single frequency across 20-20k Hz, and at the halfway point the sweep is **abruptly** punctuated by a series of "clicks" - short-term broad-band transients.

<p align="left"><img src="assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v44_split_audio.png" width="100%"></p>

### Fourier Analysis

The **Short-Time Fourier Transform** (STFT), the typical textbook approach for basic DSP, captures the *general* idea of the signal - but resolves its energy into a somewhat **smeared and jagged representation**. Low-end measurements of the sweep bleed and spread vertically into neighboring frequency bands, while the high-end appear as weak and chunky fragments in time.

<p align="left"><img src="assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v44_split_fourier.png" width="100%"></p>

### Wavelet Analysis

The **Continuous Wavelet Transform** (CWT) traces the sweep with a narrower spread and cleaner definition, resolving the burst of "clicks" as distinct events in time - arguably a more **representative** result, by capturing signal energy at the moments in time and frequency bands they actually belong to.

<p align="left"><img src="assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v44_split_wavelet.png" width="100%"></p>

### Discussion

Both produce technically valid representations of the same audio, but the difference in resolution quality comes from trying to **analyze signal properties that simultaneously exist at drastically different scales of magnitude**. This idea can be a little too abstract to immediately understand.

Think of it like trying to observe a group of trees. How exactly would you situate yourself to do this? It depends on what exactly you're observing and the range of detail you want to each achieve. To map out the entirety of the forest, you wouldn't stand up close with a microscope. To observe its plant cells, you wouldn't watch from a space-grade telescope in orbit. For every scale of detail in between these two extremes, there is a level of resolution appropriate for the detail you are observing. You would use binoculars to record individual leaf shapes, or a magnifying glass to analyze the texture of a particular kind of bark.

In any case, the observer always faces the same tradeoff - **clearly observing fine details comes at the cost of knowing exactly where these details exist in the grand scheme of themselves**. Zoomed in, you could easily figure out the kind of tree you were looking at, but would have no idea **where** in the forest you were. Zoomed out, you could easily map the acreage of the forest, but would struggle to determine **which** kind of trees make it up. 

### Comparison

The audio example and analogy above exhibit this resolution tradeoff directly. The STFT monitors the entirety of the audio using **a single lens** of resolution, so while one scale of frequency may come out looking fine, other scales skew their perception - as seen in the STFT's distortions at each end of the spectrum. The CWT is like an array of lenses spanning from microscopic to macroscopic resolution, using the appropriate lenses depending on the scale of the frequencies being measured. This achieves **multiscale resolution** for observing non-stationary, frequency-rich sound structures - simultaneously capturing the long-lasting low-ends, the short-lived high-ends, and everything in between.

> ℹ️ For more STFT vs CWT → **[DSP README](src/subshader/dsp/DSP.md)**

## Design and Performance
> 🚧 **OPEN MARKERS** — delete each block as you resolve it
> 1. **SEAM 2** — Runtime opener: derive the 186 ms per-frame deadline here now (moved from Deadline)
> 2. **SEAM 3** — bridge into the hybrid chart under Runtime (placeholder line in place)
> 3. **SEAM 4** — payoff closer: 1.4 M samples/s vs the 44.1 k intake — 31× faster than it plays

Audio processing and visual rendering are synchronized to audio playback in a multi-stage pipeline. From a file (live audio coming soon), audio samples are fetched, processed, and rendered as a time-frequency **energy spectrum plot** while in sync with the system's sound playback device. 

<p align="left"><img src="assets/timing/subshader_modules.drawio.png" width="380"></p>

**Multiscale resolution** comes at the cost of **more compute.** [NumPy](https://numpy.org/doc/stable/reference/routines.fft.html)'s STFT runs exceptionally fast, but as it was just stated, its perception can look skewed. This project uses a custom CWT adapted from [ANTS](https://www.youtube.com/playlist?list=PLn0OLiymPak2BYu--bR0ADNBJsC4kuRWs), rewritten in Python and implemented with CUDA ([CuPy](https://cupy.dev/)) to run on a GPU. The live visualization is rendered with a graphics shader for the sake of drawing megapixels worth of signal data and [matplotlib](https://matplotlib.org/) cannot handle this much detail for real-time performance.

### Documentation

→ [AUDIO README](src/subshader/audio/AUDIO.md) 

→ [DSP README](src/subshader/dsp/DSP.md) 

→ [RENDERER README](src/subshader/renderer/RENDERER.md)

### Deadline

During runtime, overlapping frames of audio samples are delivered to the DSP stages for parallel processing, results are stored in a circular buffer, and the renderer color-maps them as a rolling **energy spectrum plot** - all synced to the audio's playback.

Audio is typically recorded and played back at 44.1k samples per second, creating a natural deadline for us. The audio needs to be fetched, processed, and rendered faster than the rate it's being played at, but the CWT's complexity makes hitting this deadline difficult.

So we needed to take advantage of the independent operations of our pipeline - computations that don't need to happen in a particular order can be parallelized to happen all at the same time.

### Startup

All the heavy computation has been off-loaded to a dedicated GPU, so all the expensive setup is paid once up front - constructing the pipeline takes about 800 ms, of which roughly 80% is GPU bring-up: CUDA and OpenGL contexts, kernel and FFT-plan compilation, uploading the wavelet bank. This keeps runtime free of large allocations and transfers.

<p align="center"><img src="assets/timing/timing_startup_hybrid_v4.png" width="100%"></p>

### Runtime

> 🚧 **SEAM 2** — derive the per-frame deadline here, at the single-frame shot (moved from Deadline):
> a. Frames arrive 8,192 samples at a time - 8K ÷ 44.1K ≈ **186 ms** of audio per frame, and that's the hypothetical deadline to fetch, transform, and render it.
> b. Each frame carries 8K samples of that 44.1 kHz stream - about **186 ms** of sound - so 186 ms is the budget to fetch, process, and render one frame before falling behind.

✅ **Real-time** - each frame finishes in ~6 ms, well inside the 186 ms deadline. The slowest frame recorded was ~11 ms - still inside. Full timing report → [TIMING.md](assets/timing/TIMING.md)

<p align="center"><img src="assets/timing/timing_runtime_deadline_v6.png" width="100%"></p>

- **Deadline** - ~186 ms per chunk of audio
- **Work per frame** - ~6 ms on average - 31× room to spare
- **Resolution** - semitone bins - energy resolved at ±3% of every center frequency (116 chromatic tones · 27.5 Hz → 21.1 kHz)

> 🚧 **SEAM 3** — bridge into the hybrid chart (line below is a placeholder):
> a. Those 6 ms, stage by stage - fetch, transform, post-process, draw, and wait for the display:
> b. Zooming into one frame - five stages on their own scale:
Here's where those 6 ms go - one frame of work, stage by stage, on its own scale:

<p align="center"><img src="assets/timing/timing_runtime_hybrid_v11.png" width="100%"></p>

> 🚧 **SEAM 4** — payoff closer, calls back to the 44.1 k deadline up top (rewrite in your voice):
> a. 8,192 samples every 5.9 ms works out to about 1.4 M samples per second - the original deadline asked for 44.1 k, so the pipeline beats it 31 times over.
> b. Put another way: audio arrives at 44,100 samples per second and leaves the pipeline at roughly 1.4 million - one second of sound is transformed ~31× faster than it plays.
This project is a proof of concept written in Python, running on WSL2 - and it still clears its deadline with 31× room to spare. Plenty of room for higher resolutions or stricter deadlines.

## Real-World Implications

> 🚧⚙️🏗️ Section Under Construction 🛠️⚠️📐

Better mathematical representation doesn't just make for prettier pictures - it gives us a more representative observation and more trustworthy features. When a signal's energy is captured where it actually belongs, instead of bleeding into places it doesn't, it yields features that describe what the signal is actually doing. Pattern detection and feature extraction can only ever be as good as the representation underneath them - which is basically the whole point of this project.

This is where the fundamentals of DSP extend naturally into areas like machine learning. Typically, the relationship between the short-term details of a dataset and the long-term trends they live in is left to the models to figure out for themselves. It has even been observed that the first few layers of a convolutional neural network naturally converge towards wavelet-like filters all on their own ([AlexNet, 2012](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)).

A representation that better reflects how a signal behaves does the job of these primitive feature layers up front, so the model spends less resources trying to interpret data through a skewed perception and more on the real relationships and patterns in the data.

None of this is exclusive to audio. It's present in anything where details live at more than one size - image and video processing, sensor data, biosignal analysis, financial markets, weather modeling, anywhere patterns are being analyzed or forecasted at scale. Even within us, our eyes and ears perform a multiscale resolution of their own, naturally adapted to the real-world signals we perceive every day. Audio just happens to be a great example of detail-rich signal structures.

It contains an incredible amount of frequency information mixed into one composite signal. Details at the microscopic and macroscopic level compose the high-level features of every signal around us - forming symphonies of detail everywhere, in everything, for those with the eyes to see them.

## Future Aspirations

- [Synchrosqueezing](https://dsp.stackexchange.com/questions/71398/synchrosqueezing-wavelet-transform-explanation) - sharpening energy spectrum, tighter energy localization, distinguish beating artifacts
- [Wavelet Scattering](https://www.youtube.com/watch?v=S6LcP7txu9E) - capture modulation envelopes (tremolo vs vibrato vs beats) 
- High-Level Feature Extraction - tempo, mood, vibe, etc
- [BPM Detection](https://www.youtube.com/watch?v=FmwpkdcAXl0&t=1264s) - soft and hard onsets
- Real-Time Classification
    - Genre mapping like [everynoise.com](https://everynoise.com/) - atmospheric to dense, bouncy to smooth 
    - Color mapping to mood - happy vs sad, stress vs relief
- Stem Separation - differentiate the symphonic layers - bass, melody, percussion, vocals, etc 

## How to Install
Python 3.9+ | NVIDIA CUDA-capable GPU | OpenGL 3.3+

```bash
git clone https://github.com/eddie-water/sub-shader.git
cd sub-shader
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Run it:

```bash
python -m subshader
```