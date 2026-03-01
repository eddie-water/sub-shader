# Impetus 

- Analyzing audio signals requires methods that can identify **what** frequencies are present and **when** they occur. 

## Fourier Transform

- The Fourier Transform, one of the most useful algorithms developed for frequency analysis in modern history, is the typically approach, however, it has fundamental limitations for generic audio analysis. 
- It lacks temporal information about frequency content, and the results are opaque when it comes to reflecting any non-frequency properties of a signal
- It is intended to analyze signals with stationary properties, audio is typically abundant with non-stationarities

## Short Time Fourier Transform

- The STFT, a more advanced adaptation of the Fourier Transform addresses this limitation by localizing the measured frequencies to moments in time, but still has limitations with its time-frequency resolution

### Time-Frequency Resolution Tradeoff

- Its time-frequency resolution is fixed - its either great for measuring low frequencies but is overkill for higher frequencies or inversely, does a poor job of differentiating low frequencies, while high frequencies are measured just fine
- This is the main tradeoff STFT implementations have to decide: do you want good resolution in the time axis, or good resolution in the frequency axis
- Depending on which kinds of frequencies you are most interested in measuring, you need to choose a resolution that is best able to measure the frequencies you are interested in
- In either case, whether you are tuning the resolution to have more temporal accuracy, you take away from your ability to finely measure frequencies and vice versa
- Let's say you were measure a variety of things using a measuring tape and it only had centimeter markings, and it only has centimeter markings, you could never properly measure something that is a few millimeters
- On the other hand, if your measuring tape only had centimeters, and you were measuring something that is a few kilometers, its extremetly overkill to measure it in centimeters - you don't really need that much precision, at least not at the cost of of being unable to measure 
- Explain how low frequencies become bucketed when then


#### TF Resolution Suited for High Frequencies
- High frequencies repeat very quickly in time, so if you have poor time resolution, a frequency could occur and you wouldn't be able to measure it
- In the first case, higher frequencies are over measured, and is very computationally wasteful - theres not a lot of benefit of being able to differentiate 10,000 Hz vs 10,100 Hz (+100 Hz) because the human ear cant even differentitae those two (a 0.1 % difference in frequency)
- Where 100 Hz and 200 Hz (also +100 Hz) but the human ear can definitely hear this difference (100% difference in freqeuncy)

#### TF Resolution Suited for Low Frequencies
- Low frequencies take longer to complete cycles, so a fine time resolution is not needed here, its better to have high frequency resolution for low frequencies, since a small change in frequency at a low frequency is very noticeable percentage increase at low frequencies
- In the same vein, small changes in frequency at high frequencies aren't very noticeable to the human ear, they
- In the first case, higher frequencies are over measured, and is very computationally wasteful - theres not a lot of benefit of being able to differentiate 10,000 Hz vs 10,100 Hz (+100 Hz) because the human ear cant even differentitae those two (a 0.1 % difference in frequency)
- Where 100 Hz and 200 Hz (also +100 Hz) but the human ear can definitely hear this difference (100% difference in freqeuncy)

## Continuous Wavelet Transform 

- The CWT addresses these resolution issues with a variable time-frequency resolution (mention its actually scale which is psuedo-frequency) meaning the resolution changes proportionally with the frequency being measured at the moment
- The CWT is computationally more expensive, and conceptually more difficult to understand at first
- So, to understand the *why* the CWT addresses these issues, and to justify its computational cost requires examining what these transforms actually doing under the hood
- To do so, we'll first look at some basic vector operations.

# Foundations
- The goal is to acheive a highly accurate method for signal decomposition
- Breaking down a composite signal into its building blocks, its fundamental components
- Sort of like unmixing a can of paint - where do you even begin
- TODO Somehow tie signal decomposition to needing a way to find similarity between the shape of the signal and the shape - vector projection - which components of one vector are aligned with the other one - this is the tie - if all the components in the other align with the vector you are comparing it to, the 'similar' they are - tie this to the Dot Product Vector Projection section

## The Inner Product 

- The Inner Product is a tool for measuring similarity and is the core vehicle of the Fourier Transform and every other transform we will be looking at
- Its called the Inner Product because it 'contracts' the vectors by performing a specific set of operations on them, and the result is all caintained within a smaller 1x1 matrix aka a scalar aka a single value (as opposed to the outer product whose results is an expanded, larger matrix) 
- This format is good, give it two vectors, and the result is effecively a similarity score
- You may remember the dot product vs inner product is kinda like a square is a rectangle but a rectangle is not a square - the inner product is the more general concept
- For now, we'll talk in terms of the Dot Product, because it has a nice geometric visualization

## The Dot Product and Vector Projection 

- We stated that Dot Product takes two vectors and produces a result that indicates their similarity or alikeness - but similar in what way?
- The dot product is a measurement of how much one vector is "projected" onto another
- Another way of thinking of this is how much of one vector  casts a shadow in the same "direction* as the other 
- Geometrically this indicates how parrallel the vectors are, and the sign implies whether or not the directions match
- When vectors point the same direction, the result is large and positive
- When perpendicular, the result is zero 
- When opposite, the result is large and negative


### 2D Vectors

Simplest example. For two vectors

**a** = (a₁, a₂) and **b** = (b₁, b₂):

**a** · **b** = a₁b₁ + a₂b₂

*[Example: Two vectors on a 2D plane. .]*

### 3D Vectors

The operation extends directly to three dimensions. For two vecotrs

**a** = (a₁, a₂, a₃) and **b** = (b₁, b₂, b₃):

**a** · **b** = a₁b₁ + a₂b₂ + a₃b₃

Same principle: multiply corresponding components, sum the results.

*[Interactive: Two arrows in 3D space. Show projection and dot product value.]*

### ND Vectors
The math does not depend on three dimensions. But is a little bit harder to see since we've ran out of physical dimensions to visualize this. Just trust. For vectors with N elements:

**a** · **b** = Σᵢ aᵢbᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ

## Going from Vectors to Sequences

- At this point, "dimension" becomes misleading when you start talking about vectors with more than 3 dimensions
- It's hard to visualize something that has 100 'dimensions' because as humans we can't physically observe 100 dimensions intuitively, so we are going to change our schools of thought a moment
- Simply speaking, all a vector is is just a sequence of numbers ordered by an index. 

*[Show a sequence]*

- Just a value, followed by another one, followed by another one and so on - all of these numbers are ordered by an index
- In vector land, these "indexes" are the dimensions, but in DSP land, they're just the position of the value, where each value at a position corresponds to that index only
- Measured values at other indexes are in "other lanes" and have nothing to do with values at any other index
- From here, its more useful to think of each value at an index as an "element" is what I like to refer to them as
- Element 1 of **a** pairs with element 1 of **b**. 
- Element 99 pairs with element 99. 
- The physical interpretation of the x, y, z axes, or the i, j, k dimensions is gone - only the index correspondence matters.

*[Example: Two 10-element vectors Dot Product example.]*

## Sign Accumulation

The dot product is a signed similarity accumulator. Each element-pair multiplication contributes to the total:

| a element | b element | product | contribution |
|-----------|-----------|---------|--------------|
| positive  | positive  | positive | agreement |
| negative  | negative  | positive | agreement |
| positive  | negative  | negative | disagreement |
| negative  | positive  | negative | disagreement |
| zero      | any       | zero    | none |

The final sum reflects net agreement across all elements. Large positive: strong alignment. Large negative: strong anti-alignment. Near zero: unrelated or orthogonal.

*[Interactive: Two short vectors (5-8 elements). Color-code each product as green (positive) or red (negative). Show accumulation.]*

## Discrete Sum to Continuous Integral

The discrete dot product sums element-wise products:

Σᵢ aᵢbᵢ

For continuous functions, the analogous operation is the integral:

∫ a(t) b(t) dt

The dt represents infinitesimally small slices. The integral sums the product of both functions at every point along t. Same operation, continuous domain.

This is the inner product - the generalized form of the dot product that applies to both discrete sequences and continuous functions.