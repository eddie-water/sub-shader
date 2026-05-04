# Foundations: Wavelet Transform

## 1. Motivation

- We want to analyze audio signals with precision 
- Need to know **what** frequencies are present and **when** they occur - this is the primary motivation for finding a method for highly accurate **time-frequency** analysis 
- The standard approach is to use the **Fourier Transform**, but it has limitations in this context
- The **Wavelet Transform** is much better suited for this kind of task
- Both are built on the same foundation - **signal decomposition** 
- Beginning with simple examples, we will build up to a comprehensive and intuitive understanding of how these methods are used to decompose signals, and explore the different areas where they excel and where they fall short

---

## 2. Foundations

### 2.1 Signal Decomposition - The Goal

- The end goal is to decompose an arbitrary signal into its **fundamental components**
- In simpler terms, we want to break down a composite signal into its **basic building blocks** and see how much of each is present in the overall signal 
- This is like trying to unmix a can of paint to figure out how much of each color ingredient contributed to the overall color of the paint - where would you even begin?
- This type of problem motivates us to find a way to do two things:
    1. **Define what a signal's fundamental components are** 
    2. **Measure the presence of each component in the signal** 
- This is where the **Inner Product** comes into play - it's a general-purpose tool for measuring signal components, and we will explore these two motivations in different contexts
- Accomplishing this really depends on the signal being analyzed and the properties we're interested in measuring

### 2.2 Inner Product - The Tool

- The **Inner Product** gives us a generic way to compare a function **f** (the signal) and a reference function **g** (embodies the signal properties we want to measure) and calculate a "**similarity score**"

<div align="center">

$$
\langle \, 
\mathbf{f} \, , \, 
\mathbf{g} \,
\rangle 
$$

<em>Inner Product Notation</em>

</div>

- The result is a measurement of how **aligned** the two functions are, reflecting their similarity
- To understand how this actually works, it's helpful to see how the Inner Product operates in its simplest form: the **Dot Product**

### 2.3 Dot Product - The Basic Case

- The Inner Product is a generalization of what the **Dot Product** does to **vectors** in $\mathbb{R}^n$ 
- This may sound a little mathy, but all it means is that we are sticking to plain, regular, real numbers - no imaginary numbers (yet) or abstract numbers in weird math domains 
- As long as you can do basic **multiplication** and **addition**, it's really not too bad

<div align="center">

$$
\vec{a} \cdot\vec{b} = \sum_{i=0}^{n} a_i b_i
$$

<em>Dot Product Notation</em>
</div>

- Read this as - multiply each term in vector **a** with each term in vector **b** and sum them all up
    1. Take the first term from **a** 
    2. Take the first term from **b**
    3. **Multiply** them together
    4. Repeat 1-3 for each pair of terms
    5. **Add** up all the multiplied terms together
- Which looks like:

<div align="center">

$$
\vec{a} = 
    ( \,
    a_0 \, , \,
    a_1 \, , \,
    a_2 \, , \, 
    \ldots \, , \,
    a_n \, )
$$

</div>

<div align="center">

$$
\vec{b} = 
    ( \,
    b_0 \, , \,
    b_1 \, , \,
    b_2 \, , \, 
    \ldots \, , \,
    b_n \, )
$$

</div>

<div align="center">

$$
\vec{a} \cdot\vec{b} = 
    a_0 \cdot b_0 \, + \,
    a_1 \cdot b_1 \, + \,
    a_2 \cdot b_2 \, + \, 
    \ldots \, + \,
    a_n \cdot b_n \, 
$$

<em> Dot Product Operation </em>
</div>


- But what does this operation even really do and what does the result even really mean? 
- The result indicates how **parallel** vectors **a** and **b** are, and is the key insight to understanding how the Dot Product operation measures **alignment** or **similarity**
- The Dot Product can be geometrically visualized during **vector projection** and 

### 2.4 Vector Projection - The Geometric Interpretation

#### 2.4.1 Basic Vectors

- Every **vector** looks like an **arrow**, each having a 
    - **magnitude** - how long it is (a single / scalar value, always positive)
    - **direction** - where it points (its angle / orientation in space)

*[Figure: Basic vectors with different magnitudes and directions]*

- What does the Dot Product do?
    1. Takes a pair of vectors
    2. Performs a set of operations on them, using their magnitudes and directions
    3. Produces a scalar result 

- Geometrically, the Dot Product tells us how parallel the two vectors are
- We can all agree that a pair of **parallel** vectors appear more **similar** than a **non-parallel** pair
- Even if the parallel vectors' lengths (magnitudes) are different, the result still implies these vectors are aligned in the same direction, reflecting their alikeness

- For two vectors **a** and **b**
    - parallel + same direction = strongly similar = positive result
    - parallel + opposite direction = strongly opposite = negative result
    - perpendicular / perfectly un-parallel = no similarity / unrelated = zero result

*[Graphic: Visualize Parallel Same Direction, Parallel Opposite Direction, and Perpendicular Vectors]*

- What if these vectors were **oblique** (neither completely parallel nor completely perpendicular)?
    - oblique + any direction = potentially kind of similar or not that similar

*[Graphic: Oblique Vectors - one kind of similar, one not so similar]*

- So how does it do that exactly? 
    - Well if we took one vector's orientation, and composed the other vector using the dimensions of the first, we can see that the second vector can be composed of two basic vectors, one of them being largely parallel with the other, visually showing us why we can say the second vector is very similar to the first one, but the third one is not so similar
- This is the heart of the Dot Product and is formally called Vector Projection

#### 2.4.2 Projection Mechanics

- There are many ways to look at it, but through the lens of Vector Projection, we can see how a vector's components contribute values in the same direction as the other 
- The more a vector's component contributes 
    - **Values** in the **same direction** as the other **increases the similarity score** 
    - **Values** in the **opposite direction** as the other **decreases the similarity score**
    - **No values** in the **direction** as the other **do not affect the similarity score** 
- In all of these cases, the magnitudes of each vector will scale the Dot Product's result to be bigger or smaller (hence the name *scalar*)
- Geometrically, the Dot Product tells us how one vector is 'projected' onto the other
- This is another way saying 'How much of one vector is present in the other'
- More specifically, it tells us which of the first vector's **basic components** are **aligned** in the same direction of the other
- Almost like how much of one vector "casts its shadow" onto the other 

#### 2.4.3 From 2D to N Dimensions

- At this point, we've been saying we need a way to define the fundamental components of the *thing* we're analyzing
- What are the fundamental components of vectors? In a coordinate system like the xy plane, the components of a vector are its dimension components - the x dimension (the x axis) and the y dimension (the y axis)
- Before getting too crazy with it, we'll start with the basics: vectors with two dimensions

**2D**

- Simplest case: **a** · **b** = a₁b₁ + a₂b₂
- The components that make up these vectors are its **basis vectors** - the values along each dimension (the x and y axes)
- The dot product compares how much each component is aligned with the other

*[Visual: 2D projection]*

**3D**

- The Dot Product still works if you add a third dimension, you just continue the pattern
- Direct extension: **a** · **b** = a₁b₁ + a₂b₂ + a₃b₃
- Now three **basis vectors** (x, y, z axes), three **components** per vector

*[Visual: 3D projection]*

**ND**

- Yep, still works if you add a bunch more dimensions, even 100 for example
- It's hard to think what a 100D Vector looks like because we don't really have a way to visualize 100 dimensions
- Drop "dimension" language - think "**elements**" ordered by some kind of **index**
- Instead of the x-dimension, Element 1 pairs with element 1, element 99 pairs with element 99
- At this point, we stop calling them "components" and start calling them **elements** or **samples**
- The vector becomes a **sequence** or **array**

*[Example: Two 10-element vectors]*

### 2.5 Sign Accumulation

- Dot product is a signed similarity accumulator
- Same signs → positive contribution (agreement)
- Opposite signs → negative contribution (disagreement)
- Final sum = net agreement across all elements
- This process of measuring similarity between two sequences is called **correlation**

*[Interactive: color-coded element products]*

### 2.6 Continuous Extension

- Discrete: Σᵢ aᵢbᵢ
- Continuous: ∫ a(t) b(t) dt
- Same operation, continuous domain - the multiply-add is exactly the dot product, except now the operands are functions instead of vectors
- How much of one function's shape is present in the other
- This generalization is the inner product
- The function we compare against is called a **basis function** - a known reference **pattern** or **template**
- The choice of basis function determines what features we can detect
- The scalar result of this comparison is called a **coefficient** - it tells you "how much" of that basis function is present

---

## 3. Fourier Transform: Frequency Templates

### 3.1 The Template Question

- We have a signal (sequence of samples)
- Inner product measures similarity to... what?
- Answer: known reference patterns (basis functions)
- So what basis functions should we use?

### 3.2 Sine Waves as Templates

- What if we made the basis function a pure sine wave?
- Compare signal to a sine wave at a particular frequency
- High similarity score → that frequency is present (large **coefficient**)
- Low similarity score → that frequency is absent (small **coefficient**)
- To get a full picture, repeat for many frequencies

### 3.3 The Fourier Transform

- Inner product of signal with sine waves at every frequency
- Each frequency gets a similarity score (a **Fourier coefficient**)
- Result: frequency spectrum
- The collection of all coefficients forms the **spectrum** - a map of frequency content

### 3.4 How FFT Actually Works

- Infinite sine wave templates (no start/end)
- Accumulates similarity across entire signal duration
- Produces magnitude and phase for each frequency
- In signal processing terms, this is **correlation** between the signal and each sinusoidal template

*[Example: Sign Accumulation of a sine wave]*

### 3.5 FFT Limitations

#### 3.5.1 Temporal Information Loss

- Sine templates span entire signal
- Result: knows *what* frequencies exist, not *when*
- A note at the beginning vs end → same FFT result

#### 3.5.2 Stationarity Assumption

- Assumes signal properties don't change over time
- Music is full of non-stationarities (attacks, decays, transitions)
- FFT smears these together

#### 3.5.3 Examples

- Two close notes played sequentially vs simultaneously
- Chirp (frequency sweep)
- Transient vs sustained sounds

---

## 4. STFT: Windowed Compromise

### 4.1 Windowing Approach

- Chop signal into short segments
- Apply FFT to each segment
- Now have time information (which window) + frequency information
- The window function acts as a **filter** - it selects which portion of the signal to analyze

### 4.2 The Resolution Tradeoff

- Window size is fixed
- Short window → good time resolution, poor frequency resolution
- Long window → good frequency resolution, poor time resolution

### 4.3 Why This Tradeoff Exists

- Need enough cycles to measure a frequency accurately
- Low frequencies need longer windows (slower cycles)
- High frequencies need shorter windows (fast cycles)
- Fixed window can't serve both optimally

### 4.4 Practical Impact

- 100 Hz vs 200 Hz = 100% difference (very audible)
- 10,000 Hz vs 10,100 Hz = 1% difference (barely audible)
- Fixed resolution wastes precision where it's not needed, lacks it where it is

---

## 5. Wavelet Transform: Adaptive Resolution

### 5.1 Core Idea

- What if the "template" width varied with frequency?
- Low frequencies → wide template (good frequency resolution)
- High frequencies → narrow template (good time resolution)

### 5.2 Wavelets as Templates

- Localized oscillations (not infinite like sine waves)
- The prototype shape is called the **mother wavelet** - the base pattern before any scaling
- Scaled (stretched/compressed) versions detect different frequencies
- Still using inner product - same fundamental operation
- In implementation, the wavelet becomes the **kernel** - the pattern we slide across the signal

### 5.3 Why This Works for Audio

- Matches how human hearing perceives frequency differences
- Matches how musical information is structured (chromatic scale)
- Computational cost is higher, but results are more meaningful

### 5.4 Convolution Implementation

- To get coefficients at every time point, we slide the kernel across the signal
- At each position: compute inner product → get coefficient for that time and frequency
- This sliding inner product operation is called **convolution**
- The kernel is also called the **impulse response** - what you'd get if you fed a single spike through a filter
- Convolution with a wavelet kernel = correlation at every time point = full time-frequency map

---

## 6. Implementation Deep Dive

### 6.1 Wavelet Construction

- Gaussian envelope
- Carrier frequency
- Admissibility conditions
- Mother wavelet → daughter wavelets (scaled versions)

### 6.2 Post-Processing Pipeline

#### 6.2.1 Scale Normalization

- Why normalization is needed
- Different approaches (1/√f vs other methods)
- Impact on visualization

#### 6.2.2 Edge Effects

- Cone of Influence (COI)
- Why edges are unreliable
- Strategies for handling edge artifacts

#### 6.2.3 Magnitude Conversion

- Complex → magnitude
- Magnitude vs power (|CWT| vs |CWT|²)
- dB scaling for visualization

#### 6.2.4 Downsampling

- From full resolution to target width
- Interpolation strategies
- Preserving temporal accuracy

### 6.3 GPU Acceleration

- Memory bandwidth bottlenecks
- Ring buffer optimization
- CPU-GPU transfer reduction
- Performance benchmarks

---

## 7. Beyond Time-Frequency: Hierarchical Feature Extraction

### 7.1 The Feature Hierarchy

Understanding audio analysis requires thinking in layers of abstraction:

#### Low-Level Features (What CWT Gives You)

- **Time-frequency coefficients**: Raw CWT output matrix
- **Spectral energy distribution**: Power across frequency bands
- **Onset/offset detection**: Sharp changes in energy
- **Zero-crossing rates**: Rapid oscillations vs smooth signals
- **Spectral flux**: Change in spectrum over time

These are direct measurements from the signal - no interpretation yet.

#### Mid-Level Features (Built on CWT Output)

- **Tempo/BPM**: Periodic patterns in coefficient envelope
  - Look for regularity in low-frequency energy peaks
  - Autocorrelation of energy over time
  
- **Pitch tracking**: Frequency trajectory over time
  - Follow the dominant frequency ridge in the scalogram
  - Harmonics appear as parallel ridges
  
- **Harmonic structure**: Overtone relationships
  - Integer multiples of fundamental frequency
  - Strength and spacing of harmonics define timbre
  
- **Timbre descriptors**: Spectral shape statistics
  - Spectral centroid (brightness)
  - Spectral rolloff (bandwidth)
  - Spectral contrast (peaks vs valleys)
  - MFCC (Mel-Frequency Cepstral Coefficients)

These require aggregating and interpreting low-level features.

#### High-Level Features (Semantic Understanding)

- **Genre classification**: Rock vs Jazz vs Classical
- **Mood/emotion detection**: Happy, sad, energetic, calm
- **Speech recognition**: Phoneme patterns → words
- **Instrument identification**: Piano vs guitar vs violin
- **Music similarity**: "Sounds like..." recommendations

These require machine learning models trained on mid-level features.

### 7.2 CWT as a Feature Extractor

**Why wavelets matter for ML:**

1. **Non-stationary signal handling**
   - Music, speech, and biological signals change constantly
   - CWT captures time-varying features FFT misses
   - Essential for: onset detection, transient analysis, dynamic events

2. **Adaptive resolution**
   - Efficient feature space representation
   - Fewer coefficients needed vs raw audio samples
   - Better than fixed-resolution STFT for variable-rate phenomena

3. **Perceptual relevance**
   - Logarithmic frequency spacing matches human hearing
   - Better ML generalization - features align with perception
   - Improves classification accuracy for audio tasks

**Practical Examples:**

**Beat Tracking**
```
Audio → CWT → Extract low-freq coefficients (< 200 Hz)
      → Find periodic peaks in energy envelope
      → Estimate tempo from peak spacing
```

**Onset Detection**
```
Audio → CWT → High-freq coefficients (> 2000 Hz)
      → Compute spectral flux (frame-to-frame change)
      → Threshold crossings = note onsets
```

**Pitch Estimation**
```
Audio → CWT → Track dominant frequency ridge
      → Smooth trajectory over time
      → Output: F0 (fundamental frequency) contour
```

---

## 8. Future Directions: Machine Learning Integration

### 8.1 Classical ML Pipeline

```
Audio → CWT → Feature Engineering → ML Model → Prediction
```

**Example Workflow:**

1. **CWT coefficients** → 2D time-frequency representation
   - Input: audio waveform (e.g., 3 seconds @ 44.1kHz = 132,300 samples)
   - CWT output: (120 frequencies × 512 time bins) matrix
   
2. **Statistical features** per frequency band:
   - **Temporal statistics**: mean, variance, skewness, kurtosis
   - **Derivatives**: rate of change over time
   - **Spectral moments**: centroid, spread, rolloff, flux
   - **Energy ratios**: low/mid/high frequency balance
   
3. **Dimensionality reduction**: 
   - PCA (Principal Component Analysis): Find main variance directions
   - LDA (Linear Discriminant Analysis): Maximize class separation
   - Feature selection: Keep most informative coefficients
   
4. **Classification**: 
   - **SVM** (Support Vector Machines): Find optimal decision boundary
   - **Random Forest**: Ensemble of decision trees
   - **kNN** (k-Nearest Neighbors): Classify by similarity to training examples
   
**Use cases:** Genre classification, mood detection, speaker identification

### 8.2 Deep Learning Approaches

```
Audio → CWT → CNN/RNN → End-to-End Learning
```

**Modern Architectures:**

#### 2D CNNs on Spectrograms

- **Treat CWT output as "images"**
  - Each pixel = coefficient at (time, frequency)
  - Convolutional layers learn local time-frequency patterns
  - Pooling layers reduce dimensionality
  
- **Architecture example:**
  ```
  CWT Spectrogram (120×512)
    ↓ Conv2D (32 filters, 3×3)
    ↓ MaxPool2D (2×2)
    ↓ Conv2D (64 filters, 3×3)
    ↓ MaxPool2D (2×2)
    ↓ Flatten → Dense(128) → Dense(num_classes)
  ```

- **Applications:**
  - Music genre tagging (10-50 genres)
  - Environmental sound classification (dog bark, car horn, etc.)
  - Acoustic scene classification (park, office, street)

#### Recurrent Networks (LSTM/GRU)

- **Temporal sequence modeling**
  - Process time slices sequentially
  - Maintain memory of past context
  - Natural for evolving patterns
  
- **Architecture example:**
  ```
  CWT Spectrogram (120×512)
    ↓ Slice into 512 frames of 120 features
    ↓ LSTM(256 units, return_sequences=True)
    ↓ LSTM(128 units)
    ↓ Dense(num_classes)
  ```

- **Applications:**
  - Pitch/melody tracking over time
  - Speech recognition (phoneme sequences)
  - Music generation (predict next time slice)

#### Hybrid Models

- **CNN for spatial + RNN for temporal**
  ```
  CWT Spectrogram
    ↓ CNN: Extract frequency patterns per time frame
    ↓ RNN: Model temporal evolution of patterns
    ↓ Output: High-level prediction
  ```

- **Example: Music Transcription**
  1. CNN detects notes present at each time
  2. LSTM models note sequences
  3. Output: MIDI note events

- **Modern variants:**
  - **WaveNet**: Dilated convolutions for raw audio synthesis
  - **Transformer models**: Self-attention on time-frequency patches
  - **U-Net**: Encoder-decoder for source separation

### 8.3 Why Wavelets + Neural Networks?

**Computational Advantages:**

1. **Reduce input dimensionality**
   - Raw audio: 132,300 samples (3s @ 44.1kHz)
   - CWT: 61,440 coefficients (120 × 512)
   - ~50% reduction, even before network compression

2. **Faster convergence**
   - Networks learn from structured features, not raw samples
   - Fewer parameters needed
   - Training time reduces significantly

**Perceptual Advantages:**

3. **Meaningful representations**
   - CWT features align with human perception
   - Better generalization across datasets
   - More robust to noise and distortion

4. **Multi-scale analysis**
   - Single representation captures:
     - Transients (high frequencies, short time)
     - Sustained tones (low frequencies, long time)
   - No need for multiple parallel networks

**Practical Advantages:**

5. **Real-time capable**
   - GPU-accelerated CWT is fast (40+ FPS achievable)
   - Smaller networks = faster inference
   - Suitable for interactive applications

**Example: Speech Recognition**

```
Traditional Approach:
Raw Audio (16kHz) → MFCC (39 features) → HMM/DNN → Phonemes → Words

Wavelet Approach:
Raw Audio → CWT (5-50ms resolution) 
         → CNN (phonetic patterns) 
         → LSTM (temporal context) 
         → Phonemes → Words

Benefits:
- Captures formant transitions better (vocal tract dynamics)
- Robust to speaking rate variations (adaptive resolution)
- Fewer hand-crafted features (network learns from CWT)
```

---

## 9. Practical Applications

### 9.1 Music Information Retrieval (MIR)

**Audio Fingerprinting (Shazam-style)**
- Hash unique spectral patterns from CWT
- Create robust signatures invariant to:
  - Background noise
  - Compression artifacts
  - Tempo/pitch variations
- Database lookup for song identification

**Auto-Tagging**
- Train classifiers on CWT features for:
  - **Genre**: Rock, Jazz, Electronic, Classical, Hip-Hop
  - **Mood**: Happy, Sad, Energetic, Calm, Aggressive
  - **Instrumentation**: Vocals, Guitar, Drums, Strings
- Use cases: music library organization, playlist generation

**Recommendation Systems**
- Compute similarity metrics in wavelet space
- "Sounds like..." suggestions based on:
  - Spectral similarity (timbre)
  - Rhythmic patterns (tempo, groove)
  - Harmonic content (chord progressions)

### 9.2 Audio Production

**Source Separation**
- Isolate vocals, drums, bass from mixed tracks
- Time-frequency masking:
  1. CWT of mixed signal
  2. Identify frequency regions for each source
  3. Create masks to extract individual sources
- Applications: remixing, karaoke, sampling

**Noise Reduction**
- Adaptive filtering in wavelet domain
- Distinguish between:
  - Signal (music, speech)
  - Noise (hiss, hum, clicks)
- Threshold wavelet coefficients to remove noise

**Dynamic Range Compression**
- Frequency-dependent gain control
- Compress loud parts, boost quiet parts
- Per-band processing for natural sound

### 9.3 Health & Accessibility

**Cardiac Sound Analysis**
- Heart murmur detection from phonocardiogram
- CWT reveals:
  - S1/S2 heart sounds (normal)
  - Abnormal clicks, murmurs (pathological)
- Early detection of valve disorders

**Seizure Detection**
- EEG time-frequency patterns
- Identify seizure signatures:
  - Spike-wave discharges
  - Rhythmic oscillations
- Real-time monitoring for epilepsy patients

**Hearing Aid Signal Processing**
- Selective amplification by frequency
- Adaptive to environment:
  - Boost speech frequencies in noise
  - Compress loud transients
- Improve speech intelligibility

### 9.4 Research & Science

**Bioacoustics**
- Whale song analysis
  - Track frequency modulation patterns
  - Identify individual whales by vocalizations
- Bird call classification
  - Species identification from recordings
  - Population monitoring

**Seismology**
- Earthquake waveform analysis
- Distinguish:
  - P-waves (primary, compression)
  - S-waves (secondary, shear)
  - Surface waves
- Early warning systems

**Astronomy**
- Gravitational wave detection (LIGO)
- CWT reveals:
  - Chirp signals from black hole mergers
  - Frequency sweep as objects spiral inward
- Nobel Prize-winning application (2017)

---

## 10. Appendix: Terminology Reference

### Concept Ladder

Same ideas, different contexts:

| Stage | What you have | What you compare against | The comparison operation | The result |
|-------|---------------|--------------------------|--------------------------|------------|
| 2D/3D Vectors | vector | basis vector | dot product | component, projection |
| N-Element Vectors | sequence, array | pattern, template | dot product | similarity score |
| Discrete Signals | signal, samples | template, pattern | correlation | correlation value |
| Continuous Functions | function, waveform | basis function | inner product | coefficient |
| Fourier Analysis | signal | sinusoid, harmonic | Fourier transform | Fourier coefficient, spectrum |
| STFT | windowed signal | windowed sinusoid | windowed FFT | spectrogram bin |
| Wavelet Analysis | signal | wavelet, mother wavelet | convolution, CWT | wavelet coefficient, scalogram |
| Implementation | input | kernel, filter, impulse response | convolution | output, filtered signal |
| Machine Learning | audio | learned features | neural network | prediction, classification |

### Key Terms

**Basis vector / Basis function**  
The reference direction (vectors) or reference pattern (functions) you compare against. The "ruler" you use to measure.

**Component / Coefficient**  
The scalar result of the comparison. "How much" of the basis is present.

**Correlation**  
Measuring similarity between two signals by multiply-accumulate. Same as inner product for signals.

**Convolution**  
Correlation applied at every position - sliding the kernel across the signal.

**Kernel**  
The pattern used in convolution. Implementation term for basis function / wavelet / filter.

**Filter**  
A kernel designed to keep certain features and remove others.

**Impulse Response**  
What a system outputs when given a single spike input. Characterizes the system's behavior. Becomes the kernel in convolution.

**Mother Wavelet**  
The prototype wavelet shape at a reference scale. Scaled copies (daughter wavelets) detect different frequencies.

**Spectrum**  
The collection of Fourier coefficients. Shows frequency content without time information.

**Scalogram**  
The collection of wavelet coefficients across time and scale/frequency. The time-frequency map.

**Feature**  
A measurable property extracted from a signal. Can be low-level (spectral energy), mid-level (tempo), or high-level (genre).

**Feature Engineering**  
The process of transforming raw data into features suitable for machine learning.

**Feature Extraction**  
Using the inner product (or other operations) to measure how much of each pattern/template is present in the data.

---

## Notes for Future Development

- Add interactive visualizations for vector projection
- Include audio examples comparing FFT vs STFT vs CWT
- Provide code examples for feature extraction pipeline
- Add case studies from real-world applications
- Include links to relevant research papers
- Develop Jupyter notebooks with hands-on exercises
