# Create a virtual environment and activate 
I'm using WSL so I do:  
$ py -m venv .venv  
$ source .venv/bin/activate  

# Install the dependencies
$ py install -r requirements.txt

### If PyQt5 says something like:  
"AttributeError: module 'sipbuild.api' has no attribute 
'prepare_metadata_for_build_wheel'" 

$ pip install --upgrade pip setuptools wheel

# Run 
$ py main.py

# TODO Hierarchy
NOW > NEXT > SOON > LATER > EVENTUALLY

# Questions
Q: Why is it useful or beneficial to perform the CWT with **Complex** Morlet 
Wavelets? Specifically, what is it about them being complex helps with time
frequency analysis?

Q: What about complex signals in general? What's so good about them in the 
context of time-frequency analysis? My initial hunch is that it is related to 
Euler's formula, the one that looks like this: cos(blah) = 0.5(e^blah + e^-blah)
and how every signal has complex component, it's just usually ones you measure
have that part 0'd out. Also there I see how you can retrieve phase information
but I don't really see the how that is useful.

Q: Why is it important to know the phase of the signal? What information is
actually usefully gained from it? I've heard of Phase Key Shifting (PSK), which
is used to encode information in the phase of signals, but I don't see how that
would play a useful part in this project.

Q: How does the shape of the Gaussian used in creating the CMW affect the 
results of the wavelet based time-frequency analysis? How are the results 
affected if the Guassian rolls off steeper or shallower? 

Q: What is the significance of the Full-Width Half-Maximum and how does it 
relate to the previous question? In one of the ANTS lessons, there was an 
example where modifying it affected the steepness of the guassian and how 
'blurred' the results became, because the wider it was in the time domain, the 
more of an 'averaging' affect it had on the signal. But oppositely, the thinner 
it was in the time domain, the more energy it had in the frequency domain, so 
it was sharper at filtering things out (or maybe I have that backwards?). 
Relate this kind of intuition to the Complex Morlet Wavelet, which is a sine 
wave being manhandled by a Guassian, being used as a kernel during the CWT.
