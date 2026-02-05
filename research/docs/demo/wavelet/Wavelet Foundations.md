# Foundations: Wavelet Transform

## 1. Motivation
- We want to analyze audio signals
- Need to know **what** frequencies are present and **when** they occur - this is the primary motivation for finding a method for highly accurate **time-frequency** analysis 
- The standard approach is to use the **Fourier Transform**, but it has limitations in this context
- After a ton of research, it turns out the **Wavelet Transform** is much better suited for this kind of task
- Both transform are built upon the same foundation: the **Inner Product** which at its core is a tool for signal decomposition
- Beginning with simple examples, we will build up to a comprehensive and intuitive understanding of why the Wavelet Transform excels in the specific areas where the Fourier Transform falls short   

---

## 2. The Inner Product

### 2.1 Signal Decomposition 
- The goal is to take a composite signal and decompose it into its **fundamental components**
- In simpler terms, we want to break down the signal into its basic **building blocks** and determine how much of each block is in the overall signal
- It's like trying to unmix a can of paint to figure out how much of each color ingredient contributed to the overall color of the paint - where would you even begin?
- This type of problem motivates us to find a way to do two things:
    - **Define what a signal's fundamental components are**
    - **Measure how much of each component is present** in the signal we are analyzing
- This is where **signal decomposition** comes into play - accomplishing this effectively  depends on the kind of signal we are analyzing and the type of properties present in the signal we are interested in discovering 
- This is what the **Inner Product** allows us to do - it's a mathematical tool that produces a "similarity score" between a signal and its fundamental components 
- In the following sections, we will explore these two ideas in different contexts - starting with the **Dot Product**: the simplest form of the Inner Product, which has a nice geometric visualization

### 2.2 Dot Product and Vectors

- You typically learn about the **Dot Product** in the early chapters of 3D Vector Calculus, and even though that sounds a little daunting, as long as you know how to multiply and add numbers together, it's really not too bad

- The Inner Product is a generalization of what the Dot Product does to **vectors** in **_R_**
    - **_R_** just means the domain for all **R**eal Numbers
    - For now, we will stick to just regular numbers - none of those imaginary numbers (yet) or weird abstract numbers in weird abstract math domains... 

        *[Graphic: Inner Product Notation vs The Dot Product Notation]*

- Since we'll be using these terms a lot, brushing up on our basic calculus, every **vector** looks something like an **arrow**, each having a 
    - **magnitude** - how long it is (a single / scalar value, always positive)
    - **direction** - where it points (its angle / orientation in space)
        
        *[Graphic: Show some vectors, big, small, and tall in different directions]*

- What does the Dot Product do?
    1. Takes a pair of vectors
    2. Performs a set of operations on them, using each vector's magnitude and direction
    3. Produces a scalar result, which again, is just a singular value

- Geometric Interpretation 
    - Geometrically, the Dot Product tells us how parallel the two vectors are
    - We can all agree that a pair of **parallel** vectors appear more **similar** than a **non-parallel** pair
    - Even if the parallel vectors' lengths (magnitudes) are different, the result still implies these vectors are aligned in the same direction, reflecting their alikeness

    - Say we have two vectors **u** and **v**
        - parallel + same direction = strongly similar = positive result
        - parallel + opposite direction = strongly opposite = negative result
        - perpendicular + any direction = no similarity / unrelated = zero result

        *[Graphic: Visualize Parallel Same Direction, Parallel Opposite Direction, and Perpindicular Vectors]*

    - What if these vectors were **oblique** (neither completely parallel nor completely perpindicular)?
        - oblique + any direction = potentially kind of similar or not that similar

        *[Graphic: Oblique Vectors - one kind of similar, one not so similar]*

    - In all of these cases, the magnitudes of each vector will scale the Dot Product's result to be bigger or smaller (hence the name *scalar*)

    - So how does it do that exactly? $ TODO sentence setting up the introduction for Vector Projection - the oblique scenario where we said okay its kind of similar

### 2.3 Vector Projection
- Geometrically, the Dot Product tells us how one vector is 'projected' onto the other
- This is another way saying 'How much of one vector is present in the other'
- More specifically, it tells us which of the first vector's **basic components** are **aligned** in the same direction of the other
- It shows us which component of the first vector is parallel with the othe
- Almost like how much of one vector "casts its shadow" onto the other 
- There's a bunch of different ways to look at it, but this is the main vehicle ($ TODO find better word for this) the Dot Product measures similarity ($ TODO I feel like I'm being circular here, but I want to tie vector projection to parallelism and the dot product)

#### 2.3.1 Defining the Fundamental Components of Simple Vectors
- At this point, we've been saying we need a way to define the fundamental components of the *thing* we're analyzing
- What are the fundamental components of vectors? In a coordinate system like the xy plane, the components of a vector are its dimension components - the x dimension (the x axis) and the y dimension (the y axis)
-
- Before getting too crazy with it, we'll start with the basics: vectors with two dimensions

#### 2.3.1 2D Vectors

- Simplest case: **a** · **b** = a₁b₁ + a₂b₂
- Each vector has **components** - the values along each **basis vector** (the x and y axes)
- The dot product compares how much the components align

*[Visual: 2D projection]*

#### 2.3.2 3D Vectors

Going from components to dimensions
- x dimension component, y dimension component, z dimension component
- 
- Direct extension: **a** · **b** = a₁b₁ + a₂b₂ + a₃b₃
- Now three **basis vectors** (x, y, z axes), three **components** per vector

*[Visual: 3D projection]*

#### 2.3.3 ND Vectors - Going From  Dimension to Elements (Vectors → Sequences)
- Math works for any N
- Drop "dimension" language - think "elements" ordered by index
- Element 1 pairs with element 1, element 99 pairs with element 99
- At this point, we stop calling them "components" and start calling them **elements** or **samples**
- The vector becomes a **sequence** or **array**

*[Example: Two 10-element vectors]*

### 2.6 Sign Accumulation
- Dot product is a signed similarity accumulator
- Same signs → positive contribution (agreement)
- Opposite signs → negative contribution (disagreement)
- Final sum = net agreement across all elements
- This process of measuring similarity between two sequences is called **correlation**

*[Interactive: color-coded element products]*

### 2.7 Discrete Sum → Continuous Integral
- Discrete: Σᵢ aᵢbᵢ
- Continuous: ∫ a(t) b(t) dt
- Same operation, continuous domain - the multiply-add is exactly the dot product, except now the operands are functions instead of vectors
- How much of one function's shape is present in the other
- This generalization is the inner product
- The function we compare against is called a **basis function** - a known reference **pattern** or **template**
- The choice of basis function determines what features we can detect
- The scalar result of this comparison is called a **coefficient** - it tells you "how much" of that basis function is present

### 2.8 Inner vs Outer Product (Brief Note)
- Inner product: contracts (rank 1 × rank 1 → rank 0)
- Outer product: expands (rank 1 × rank 1 → rank 2)
- Named relative to operand rank - inner goes inward toward scalars

---

## 3. Applying the Inner Product: What Do We Compare Against?

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

---

## 4. FFT Limitations

### 4.1 Temporal Information Loss
- Sine templates span entire signal
- Result: knows *what* frequencies exist, not *when*
- A note at the beginning vs end → same FFT result

### 4.2 Stationarity Assumption
- Assumes signal properties don't change over time
- Music is full of non-stationarities (attacks, decays, transitions)
- FFT smears these together

### 4.3 Examples
- Two close notes played sequentially vs simultaneously
- Chirp (frequency sweep)
- Transient vs sustained sounds

---

## 5. STFT: A Partial Fix

### 5.1 Windowing Approach
- Chop signal into short segments
- Apply FFT to each segment
- Now have time information (which window) + frequency information
- The window function acts as a **filter** - it selects which portion of the signal to analyze

### 5.2 The Resolution Tradeoff
- Window size is fixed
- Short window → good time resolution, poor frequency resolution
- Long window → good frequency resolution, poor time resolution

### 5.3 Why This Tradeoff Exists
- Need enough cycles to measure a frequency accurately
- Low frequencies need longer windows (slower cycles)
- High frequencies need shorter windows (fast cycles)
- Fixed window can't serve both optimally

### 5.4 Practical Impact
- 100 Hz vs 200 Hz = 100% difference (very audible)
- 10,000 Hz vs 10,100 Hz = 1% difference (barely audible)
- Fixed resolution wastes precision where it's not needed, lacks it where it is

---

## 6. CWT: Variable Resolution

### 6.1 Core Idea
- What if the "template" width varied with frequency?
- Low frequencies → wide template (good frequency resolution)
- High frequencies → narrow template (good time resolution)

### 6.2 Wavelets as Templates
- Localized oscillations (not infinite like sine waves)
- The prototype shape is called the **mother wavelet** - the base pattern before any scaling
- Scaled (stretched/compressed) versions detect different frequencies
- Still using inner product - same fundamental operation
- In implementation, the wavelet becomes the **kernel** - the pattern we slide across the signal

### 6.3 Why This Works for Audio
- Matches how human hearing perceives frequency differences
- Matches how musical information is structured
- Computational cost is higher, but results are more meaningful

### 6.4 Convolution: Sliding the Kernel
- To get coefficients at every time point, we slide the kernel across the signal
- At each position: compute inner product → get coefficient for that time and frequency
- This sliding inner product operation is called **convolution**
- The kernel is also called the **impulse response** - what you'd get if you fed a single spike through a filter
- Convolution with a wavelet kernel = correlation at every time point = full time-frequency map

---

## 7. Wavelet Construction (Future Section)
- Gaussian envelope
- Carrier frequency
- Scaling relationship
- Mother wavelet → daughter wavelets (scaled versions)

## 8. Post-Processing Pipeline (Future Section)
- Scale normalization
- Edge effects
- Complex → magnitude
- Downsampling

---

## Appendix: Terminology Reference

A concept ladder - same ideas, different contexts:

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