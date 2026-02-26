# Why Not FFT?

The FFT is one of the most clever and useful algorithms developed in modern histroy. It's the typical solution for analyzing signals for their frequency content. However, shortly after a trivial attempt, I noticed that the results were pretty far off of my expectations. I tried using it for this application.  audio analysis, I ran into many of its main limitations pretty quickly. The FFT becomes opaque for musical signals - it can't tell you *when* frequencies are present, only *that* they're present. The wavelet transform is much better suited for this task. But before we get there, let's explore the foundation: the Inner Product.

## The Inner Product

In Calc 1 you learn the derivative, the integral. In vector calculus you get the dot product and cross product. We're diving into the dot product, which is a type of Inner Product.

The dot product takes two vectors and produces a scalar - fancy way of saying you take two sequences of data and get a single value out. What does that value represent? It's basically a similarity score. It tells you how much of one vector lies in the same "direction" as the other.

*[Examples of dot product projection stuff]*

## From Dimensions to Elements

In vector calculus, vectors typically represent physical dimensions: x, y, z. But the math works beyond 3D. The key insight: each vector element doesn't have to correspond to a physical dimension. Think of vectors simply as sequences of numbers ordered by index. The index *is* the dimension.

A value in the x dimension tells you nothing about y or z - obvious for 3D vectors, but important to remember when generalizing. If you have two vectors with 10 elements each, it's not "x dimension from v1 plus x dimension from v2" - it's element 1 from v1 plus element 1 from v2, element 10 from v1 plus element 10 from v2.

This lets us generalize the dot product into the Inner Product.

## How Does Multiplying and Adding Equal Similarity?

The dot product is effectively a **signed similarity accumulator**. If two corresponding elements have the same sign (both positive or both negative), they contribute positively to the similarity score. Opposite signs contribute negatively. Zeros contribute nothing.

Think about it: if one signal is going up (positive) and the other is also going up (positive), that implies alignment. If one goes up while the other goes down, they're anti-aligned. The dot product accumulates all these signed contributions across every element.

*[Example]*

## Connection to Fourier

Look at the Fourier transform definition - you see an integral of a signal multiplied by some weird exponential term. The integral is just a sum of a bunch of terms. What's happening is we're multiplying corresponding elements and adding them up.

That's exactly what the dot product does.

---

*Disclaimer: I'm not a math scientist - just an EE with calc, diff eq, linear algebra, and some DSP classes. This is conversational-level explanation. I probably have some assumptions wrong or use improper terms, but for understanding the concept, this should work.*