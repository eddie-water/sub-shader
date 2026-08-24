# SubShader

SubShader is a **real-time audio visualizer** written in Python. It uses modern techniques in digital signal processing and GPU programming to accurately **visualize what an audio signal sounds like**. But to call it just a visualizer is a massive understatement - it implements the foundation of a GPU-accelerated pipeline customly built for real-time signal processing and feature extraction.

This project uses **[Wavelet](https://youtu.be/jnxqHcObNK4?si=x98elLTbz6QLe03g&t=1996)**-based analysis to convert audio signals into a **time-frequency** representation for visualization. Wavelet analysis is a modern adaptation of the traditional **[Fourier](https://youtu.be/spUNpyF58BY?si=jXTsOaIHUwB8meoc)**-based methods used for typical signal analysis. The advantages and justification of using it for real-world signal processing are discussed in this project.

> ℹ️ For in-depth implementation decisions and design intuitions → **[DSP.MD](src/subshader/dsp/DSP.md)** 

Demonstrating my technical skills in real-time signal analysis and GPU acceleration, I'm using this project to branch into areas more adjacent to digital signal processing (DSP), machine learning (ML), data science, and computer engineering. The high-level design, performance, real-world applications, and future aspirations of this project are detailed below. **It took a lot of effort and care to make this, so thank you for taking the time to read!**

<p align="left"><img src="assets/images/readme/beltran_scalogram_placeholder_v1.png" width="100%"></p>
🚧 TODO - Replace with live audio clip 🚧

## Problem Statement

When we perceive sound, our ears can easily distinguish all the different kinds of noises we hear - the loud vs the quiet, the high vs the low, the sudden vs the gradual. However, designing something that can analyze audio to this level of detail is not as trivial as it may seem. To properly visualize an audio signal, we need to be able to detect more than just **what frequencies** are present. We also need to know **when in time** they start, how long they last, and when they stop. All relative to how loud these frequencies actually are to each other. The main difficulty of this stems from trying to analyze sound structures and **details that coexist at vastly different scales**. 

Typical audio like music or speech is frequency-rich, containing both low and high-end frequencies that contribute to the long-term trends and short-term details of a signal's structure. High-frequency activities can be brief and abrupt, demanding **fine resolution in time** to pin down exactly when they happen. Low-frequency contributions take long stretches of time to develop, shaping the grand scheme of the signal slowly and gradually - demanding **fine resolution in frequency** instead to detect slight variations in tone. Simultaneously capturing both ends of a signal's frequency spectrum in a fully-detailed representation is the core problem this project confronts.

### Audio Example

The audio displayed here is a "non-stationary" signal containing both regular and irregular frequency content, and is an exaggeration of this resolution problem specifically. The signal **gently** sweeps a single frequency across a range of 20-20k Hz - from the middle, down to the low-end, and back up to the high-end. At its halfway point, the sweep is **abruptly** punctuated by a series of "clicks" or "short-term broad-band transients". 

<p align="left"><img src="assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v44_split_audio.png" width="100%"></p>

### Fourier Analysis

The **Short-Time Fourier Transform** (STFT) is the typical, textbook-approach for basic DSP. It does a decent job of representing the *general* idea of what is happening in the signal. But for "non-stationary" frequency content, notice how the signal's energy is resolved into a somewhat **smeared and jagged representation**. Notice how low-end frequency measurements of the sweep bleed and blob into neighboring frequency bands, while high-frequency impulses like the clicks appear as weakly represented fragments.

<p align="left"><img src="assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v44_split_fourier.png" width="100%"></p>

### Wavelet Analysis

The **Continuous Wavelet Transform** (CWT) is the more-recently popularized method for advanced signal analysis. It tracks the frequency sweep much more closely, tracing it with clean definition and without much smudging. It clearly resolves the onset of each "click" as a distinct event in time. Overall, the implementation produces an arguably **more representative result** by capturing signal energy into the frequency bands and moments in time they actually belong to.

<p align="left"><img src="assets/images/dsp/figures/by_figure/fig_1_fourier_vs_wavelet/fig_1_fourier_vs_wavelet_hero_v44_split_wavelet.png" width="100%"></p>

### Discussion and Intuition

These methods produce two technically valid, but slightly different representations of the same audio. We can clearly see the CWT excels at perceiving signal energy in the bands and moments they actually occur. The STFT's slightly skewed perception at each end of the spectrum is its main limitation for analyzing audio - signals that commonly exhibit this kind of frequency content. But what is the specific issue faced by the STFT? Like previously mentioned, it stems from trying to **analyze signal properties that simultaneously exist at drastically different scales of magnitude.**

But this idea can be a little too abstract to immediately understand - what counts as a detail, and the resolution it needs, depends entirely on the thing being observed. Think of it like trying to observe a group of trees. If you were trying to map out the entirety of the forest, you wouldn't stand up close with a microscope. If you were trying to observe its plant cells, you wouldn't watch from a space-grade telescope in orbit. For every scale of the detail measured at each of these extremes, there is an appropriate level of resolution needed to represent it meaningfully. What about the scales of detail in between each of these extremes? What kind of resolution would they need? You would probably use binoculars if you were trying to record the individual leaf shapes on a specific kind of tree. If you were trying to analyze the texture of a particular kind of bark, you might use a magnifying glass.

In any of these cases, the observer is faced with the same tradeoff - **being able to observe fine details clearly comes at the cost of knowing exactly where these details exist in the grand scheme**. Zoomed in, you could easily figure out what kind of tree you are looking at, but you would have no idea **where** in the forest you were. Zoomed out, you could easily map the acreage of the forest, but would struggle to determine **which** specific types of trees make it up.

### Method Comparison

The audio example shown above exhibits this resolution tradeoff analogy directly. The STFT tries to monitor the entirety of the audio using **a single lens** of resolution - meaning one scale of frequency may come out looking fine, but all other scales come out skewed. The CWT is like an array of lenses spanning from microscopic to macroscopic resolution - except here we are not mapping the acreage of a forest, or recording leaf shapes, we are observing detail-rich sound structures. We are resolving the signal at a level of detail that allows us to distinguish its long-lasting, low-end frequencies from its short-lived, high-end ones, and everything in between.

> ℹ️ All design decisions related to this concept are explored in depth in **[DSP.md](src/subshader/dsp/DSP.md)**

## Design

Better resolution comes at the cost of more compute. Libraries to implement the STFT like [NumPy](https://numpy.org/doc/stable/reference/routines.fft.html) run exceptionally fast, but produce the smeared resolution issue shown above. [PyWavelets](https://pywavelets.readthedocs.io/en/latest/), a popular wavelet library, produces clean results but is too slow for real-time visualization. This project uses a custom CWT adapted from [ANTS](https://www.youtube.com/playlist?list=PLn0OLiymPak2BYu--bR0ADNBJsC4kuRWs), rewritten in Python and parallelized on a GPU with [CuPy](https://cupy.dev/). The renderer is a graphics shader for the same reason - [matplotlib](https://matplotlib.org/) cannot draw frames this detailed at this rate, and downsampling them would throw away detail the CWT strives to capture.

### Pipeline
[AUDIO.MD](src/subshader/audio/AUDIO.md) | [DSP.MD](src/subshader/dsp/DSP.md) | [RENDERER.MD](src/subshader/renderer/RENDERER.md)

The pipeline processes and visualizes audio signals in sync with audio playback. It fetches audio samples from a file (live audio coming soon), processes them, and renders the results as an energy spectrum plot. The pipeline is composed of the following stages:

<p align="left"><img src="assets/timing/subshader_modules.drawio.png" width="380"></p>

During runtime, overlapping frames of audio samples are delivered to the DSP stages for parallel processing. The CWT is performed on the audio frame and the results are post-processed. After storing the most recent frame into a circular buffer, the renderer uses a shader to color-map the results as a rolling energy spectrum plot. All of this is synced to the audio's playback device.

<p align="left"><img src="assets/timing/subshader_runtime.drawio.png" width="1000"></p>

## Performance

✅ **Real-time** - each frame finishes in ~80 ms, well inside the 185 ms deadline. The slowest frame recorded was ~130 ms - still inside. Full timing report → [TIMING.md](assets/timing/TIMING.md)

<p align="center"><img src="assets/timing/timing_runtime_gantt_black_bg_v1.png" width="100%"></p>

- **Deadline** - ~185 ms per chunk of audio
- **Work per frame** - ~80 ms on average - 2.3× room to spare
- **Resolution** - semitone bins - energy resolved at ±3% of every center frequency (116 chromatic tones · 27.5 Hz → 21.1 kHz)

### Startup
To achieve real-time performance, all the heavy DSP computation and rendering are off-loaded to a dedicated GPU, and all of the expensive setup is paid once up front. Constructing the pipeline takes about 1.3 s, of which roughly 80% is GPU bring-up - creating the CUDA and OpenGL contexts, compiling kernels and FFT plans, and uploading the wavelet bank. This minimizes large memory allocations and transfers during runtime.

<p align="left"><img src="assets/timing/subshader_startup.drawio.png" width="650"></p>
<p align="center"><img src="assets/timing/timing_startup_gantt_black_bg_v1.png" width="100%"></p>

This project is a proof of concept written in Python, running on WSL2. Most of the overhead is from my environment, not math. When the core stages were benchmarked in isolation, the same process was about 8× faster - for a potential 18× margin. Plenty of room for higher resolutions or stricter deadlines.
<!-- SKELETON R3: environment-vs-isolation is its own beat - WSL2 proof-of-concept
numbers vs isolated core-stage benchmark, framed as a deliberate contrast / headroom,
two numbers side by side, no hedging, not an apology. -->

## Real-World Implications

> ### 🚧⚙️🏗️ Section Under Construction 🛠️⚠️📐
>
> **Goal for this section:** Close the loop - this is everywhere in the natural world: the signals around us, and the ones in our ears and eyes. Better mathematical representation doesn't just make for prettier pictures - it gives us a more representative observation. We are tuning our measurement design to adhere to the same natural law the signal we are measuring abides by - the same we can't know both where an electron is and how fast it's moving - is the same as the tree analogy sort of - here we are controlling this that observational tradeoff to our advantage. Better representation has massive benefits downstream: real-time data analysis and machine learning - classification, mood detection, embedding selection, model fine-tuning - and triggering real-time visuals more accurately. Beyond audio too - graphics, small sample machine learning, financial time series

<details>
<summary>📝 Consolidate the following ideas into something conclusive </summary>

- Better mathematical representation doesn't just make for prettier pictures
- A signal's energy captured where it actually belongs, instead of bleeding into places it doesn't → more reliable features that are actually representative of the signal's properties
- Pattern detection and feature extraction can only ever be as good as the representation underneath them - the fundamentals of DSP explored here extend naturally into machine learning
- "...delivered to a machine learning model as higher quality inputs for it to learn and derive patterns from" (the payoff line - weave into final prose)
- None of this is exclusive to audio - almost every signal in the natural world is composed of both gradual trends and fine details
- Multiresolution analysis applies to any domain where details live at more than one size
- Wavelets follow the same natural laws as the signals they are measuring
- Typically this is left to the models to figure out - how the quickly-changing parts of a signal exist in the same context as the slow-moving ones
- The first layers of convolutional neural networks tend to converge towards wavelet-like filters ([AlexNet, 2012](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)) - even self-optimizing systems gravitate toward this innate time-frequency compromise
- Even our own eyes and ears run a multiscale decomposition of the world
- The cochlea resolves low tones with narrow biological filters and high ones with wide ones - the same proportional trade the wavelet makes
- Our eyes can take in the landscape trends of our view while simultaneously focusing on the finer details of an object in sight; our ears can distinguish the texture of a sound as its frequency content changes with time
- It is valuable to accurately capture the short term details in the context in which they exist - it is happening around us constantly
- ear -> graphics card -> eye 
- there are symphonies everywhere, for those with the eyes to see them

</details>

## Future Aspirations
<details>
<summary>📝 Speak on the future direction of the project </summary>

- **Synchrosqueezing** - sharpening the CWT's energy localization even further 
   - What about beats? 
   - Consider two signals at +/- X Hz at Y Hz - this creates a X Hz beating artifact
   - What contexts is this desired? Simple monotonic analysis or rich, complex, irregular polyphonics? 
- **Wavelet scattering** - capturing modulation envelopes
   - Differentiating tremolo vs vibrato vs beats
- **Tempo/BPM detection** - tempo tracking that holds up even with soft onsets, driven by energy detection 🚧 TODO link tempo algorithm video 🚧
- **Real-time classification + tempo tracking** - triggering visual stimulus based on actual signal events, more accurately
- **Genre detection** - mapping audio along axes like [everynoise.com](https://everynoise.com/) - atmospheric to dense, bouncy to smooth
- **Color mapping to mood** - happy chords vs sad chords, chords that stress each other or relieve each other
- **Stem separation** - differentiating the parts in the audio to stem things out
</details>

## How to Install
Requirements:
- Python 3.9+
- NVIDIA CUDA-capable GPU
- OpenGL 3.3+

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
