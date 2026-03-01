The goal of this section is to explain the impetus for a better time frequency analysis method. Typically people use the FFT which is one of the most beautiful and clever algorithm which came out like 200 years ago. However, after trying to use it, I ran into a lot of its pitfalls for an applicatoin like music analysis.

TLDR the FFT is not really intended to anaylyze musical signals, something like the wavelet trasnform is much better suited for such a task. The Fourier Transform becomes opaque, and cannot reflect the properties of the signals we are interested. Turns out, the wavelet transform is much at determined when in time, frequencies are present in the composite signal. But before we get to there, we will explore why you want to use any kind of transform to analyze music.

The Inner Product
It all starts with the Inner Product. In Calc 1 you learn of some fundamental tools like the derivitive, the integral, and as you progress to more intermediate calculus, like vector calculus, you learn more tools like the dot product and cross product. We are going to dive into the dot product, which is a type of Inner Product.

You might remember that the dot product takes in two vectors, and produces a scalar result. What even is that. Fancy way of saying you take two sequences of data, and it produces a single value. What does that value represent? It vasically gives a score of how 'similar' those two vectors are. Does that ring a bell? It basically tells you how much of one vector is in the same dimension as the other.

Examples of the dot product projection stuff.

So how does this relate to signals and frequencies and how can it be used for musical analysis? Doesn't seems like multidimensional vector math to me. But thats because the vector dimension stuff is just an applicaiton of the Inner Prodcut at work. Its a more general dot product.

In vector calculus, the vectors represent physical (or even abstract) dimensions. Typically its the x, y, and z dimensions. This is nice because we can visualize this in a 3D coordinate system. But the math works beyond 3D vectors.

Going from dimensions to elements.

The Inner Product doesn't necessarily only cover 3D vectors. The thing to note here is the each vector has 3 elements in them. The first belonging to the x, the second belonging to y, and the third corresponding to z (typically). But if there was anothe dimension, you could still throw it in there and the result would simply account for this new mysterious fourth dimension. Here's the part we need to get past. The elements do not have to correspond to physical dimension. We need to think about these vectors simply as sequences of numbers that are ordered by some sort of index. The index, or position of the element is its 'dimension'. So for whatever structure the vector is, its important to know that the value of descrive element, does not give any information about any of the others. 

A value for the x dimension does not indicate anything about the values of the y or z dimension. This is like a duh for 3D vectors. But for generic vectors its important to keep this in mind. If you add two vectors together, you add the values of element one in each vector together, then the values in element two together, and so on and the result is a new super-positioned vector where it doesn't matter if you add the first vector to the second vs adding the second vector to the first (this is super important esp when we talk about LTI systems later)

Let's say we two vectors each with 10 elements. Instead of thinking of them as 'dimensions' we just need to think about each element being a value that corresponds to its own position in the vector. So it's not x dimension from v1 plut x dimension from v2, it's element 1 from v1 and elemnt 1 from v2. Element 10 from v1 plus element 10 from v2. Its not too terrible to think in this way.

So now that we think of these dimensions in terms of elements, we can generalize the dot product into the more general Inner Product

So what's the point? I'm going to use the term dot product interchangeably but they are different. For now we will say the dot product is the discrete operation on the values, and the inner product is the more general term.

So what does the dot product do? It measures 'simlariy' between two vectors. How does it do that? What does 'similar' mean in this context? What it does is called vector projection. Projection in simplespeak is seeing how much of one thing lies within the other. In vector calc, its seeing how much of each dimension. So if we have a vector where theyre idenitcal it pretty obicious too see how both elemtns in each vector are identical, and that  oe vectro copletel proojects itself onto the other. But as you add more dimesnions, and different sizess,  it ells you ow much the directiona ligns, and then how much the magnitudes contribute. A clear example of no projeion is a vector with values in the c dimesinon, and nothingin the y dimension, and then aother vector nothing in the x and somehting in the uy here it obvious, but if we compute it we will see since the first has no cotribution in they dimension and the seond does, and the seond has no contribution in the x dimnsion, abb and the scond one has no contributin form the x, we can see these two vector dont project at all onto each other.

the result is zero. remeerb the dot product gives you a scalar result, so the its almost like a similaryt score. how can this applied to more dimesnions? Well it just does. in the 10 element vecors, conribution from the first elemtn are compare to contributions of the first elemtn in the seond vector and so on.

For me this was hard to undertand how thefourier trasnform does this- well if you look at the definition of the fourier trasnform you see the interfral of a signal and this weird term here. Let's not worry about what they are right now, lets just say they are both just numbers in a seqeunce that can be indexed by position. The integral is just a fancy way of saying the sum of a bunch of terms. The dt you can think of as just the tiny bits that make the entire function. So whats happening here is we are multiplying each element in both vectors and then adding them up together.

News flash, thats exactly what the dot product does. It takes eah elemtn, multiply them together and thne addd them all up. This is all fine and dandy, but it still doesn't make the connection for me regarding the simlairy aspect. Howe does multiply and adding things togther equal similarity? Next topic, sign accumulation.

Sign accumulation
The dot product is effectively a sign accumulator. We are going to say that if two points have the same sign, they 'correspond' to each other, and if they have opposite signs, or one/both are zero, they do not correspond. The correspondanc eof these points contribute to the overall simlaiy score. Think about it. If a point is positive, and the other is also positive, those two points imply some simliarity in the two vectors. Since each element only corresponds (need a different word here because this kind of corresponding is referring to the dimension corresponding concept) to each othe relemtns in its position, we need to consider all the points next to them too. We dont know if they points pair before them are pulling the signal up or dowm. Also each point has equal weight in the over all score. One leemtn being similar is not more imporant to the contributoin than its neighbors, they all can contribte less or more.

Example

Anyways,`w

Disclaimer, I'm not a math scientist. I didn't take any formal math classes besides the ones in my electrical engineering classes like the basic calc classes, some dif eq, some linear algebra and some DSP classes (but those were more  practical implementaiton classes and not too too theory based), so i'm providing a converstaional level of explanation. I probably have some assumption wrong, or will use improper terms, but for the sake of understanding the concept this