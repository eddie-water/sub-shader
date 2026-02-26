# DSP Documentation Structure v2

## 1. Hook (Brief)
- We want to analyze audio signals
- Need to know **what** frequencies are present and **when** they occur
- This requires time-frequency analysis
- The standard tool is the FFT, but it has limitations → CWT is better suited
- To understand *why*, we need to look at what these transforms actually do
- Starts with basic vector operations

---

## 2. Foundations: The Inner Product

### 2.1 Signal Decomposition Goal
- Breaking down a composite signal into its building blocks aka its fundamental components 
- Like unmixing a can of paint
- Need a way to measure "how much" of each building block exists in a signal = "how similar" to each of its building block patterns
- This is what the inner product does - it effectively produces a similarity score between two things which measures their alikeness
- We'll start with the Dot Product, the simplest Inner Product for vectors, which has a nice geometric visualization

### 2.2 The Dot Product and Vector Projection
- Dot product: takes two vectors, produces a scalar
- Result indicates similarity - how much one vector projects onto another
- Geometric interpretation: parallel = large positive, perpendicular = zero, opposite = large negative

### 2.3 2D Vectors
- Simplest case: **a** · **b** = a₁b₁ + a₂b₂

*[Visual: 2D projection]*

### 2.4 3D Vectors
- Direct extension: **a** · **b** = a₁b₁ + a₂b₂ + a₃b₃

*[Visual: 3D projection]*

### 2.5 N Elements (Vectors → Sequences)
- Math works for any N
- Drop "dimension" language - think "elements" ordered by index
- Element 1 pairs with element 1, element 99 pairs with element 99

*[Example: Two 10-element vectors]*

### 2.6 Sign Accumulation
- Dot product is a signed similarity accumulator
- Same signs → positive contribution (agreement)
- Opposite signs → negative contribution (disagreement)
- Final sum = net agreement across all elements

*[Interactive: color-coded element products]*

### 2.7 Discrete Sum → Continuous Integral
- Discrete: Σᵢ aᵢbᵢ
- Continuous: ∫ a(t) b(t) dt
- Same operation, continuous domain - notice how this multiply-add operation is exactly what the dot product, except now instead of the basis being vectors, the basis are functions 
- You can think of it at this point, how much of one function's shape is present in the other
- We can make these basis functions any shape or pattern we want!
- What if we made this basis functions a pure sine waves? Then we could take a function and compre it to a sine wave that oscillates at a particular frequency, and the result will produce a similarity score between the function and the sine wave (need to introduce the word basis function earlier at some point so here we can make the connection, but mayve we put this in the below section)
- This generalization is the inner product


### 2.8 Inner vs Outer Product (Brief Note)
- Inner product: contracts (rank 1 × rank 1 → rank 0)
- Outer product: expands (rank 1 × rank 1 → rank 2)
- Named relative to operand rank - inner goes inward toward scalars

---

## 3. Applying the Inner Product: What Do We Compare Against?

### 3.1 The Template Question
- We have a signal (sequence of samples)
- Inner product measures similarity to... what?
- Answer: known reference patterns (templates)

### 3.2 Sine Waves as Templates
- To detect a frequency, compare signal against a sine wave of that frequency
- High similarity score → that frequency is present
- Low similarity score → that frequency is absent

### 3.3 The Fourier Transform
- Inner product of signal with sine waves at every frequency
- Each frequency gets a similarity score
- Result: frequency spectrum

### 3.4 How FFT Actually Works
- Infinite sine wave templates (no start/end)
- Accumulates similarity across entire signal duration
- Produces magnitude and phase for each frequency

*[Example Sign Accumulation of a sine wave]*
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
- Scaled versions for different frequencies
- Still using inner product - same fundamental operation

### 6.3 Why This Works for Audio
- Matches how human hearing perceives frequency differences
- Matches how musical information is structured
- Computational cost is higher, but results are more meaningful

---

## 7. Wavelet Construction (Future Section)
- Gaussian envelope
- Carrier frequency
- Scaling relationship
- Convolution as sliding inner product

## 8. Post-Processing Pipeline (Future Section)
- Scale normalization
- Edge effects
- Complex → magnitude
- Downsampling