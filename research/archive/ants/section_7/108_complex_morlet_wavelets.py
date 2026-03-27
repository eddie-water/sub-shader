import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

pi = np.pi

sample_rate = 1000
t = np.arange(-1, 1, (1 / sample_rate))
freq = 6 # The guy in the video uses 2pi as example frequency which is dumb

sine_w = np.exp(1j*2*pi*freq*t)

# plt.plot(sine_w.real)
# plt.plot(sine_w.imag)
# plt.show()

fwhm = 0.5
gaus_window = np.exp(-4*np.log(2)*t**2 / fwhm**2)

cmw = sine_w * gaus_window

plt.plot(t, cmw.real,'b', label='real part')
plt.plot(t, cmw.imag,'r--', label='imag part')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Complex Morlet wavelet in the time domain')
plt.show()

num_points = len(t)
cmw_x = abs(fft(cmw) / num_points)

hz = np.linspace(0, sample_rate, num_points)

'''
Something I don't really understand:
So the x axis from the linspace call above producing the plot below is just a
shortcut and not really representative of what the axis really looks like. It 
is supposed to be the first half of the axis is positive, and the second half
are the negative versions? He is saying: the axis is supposed to look like
0, 1, ..., 499, 500 (the Nyquist), ..., -400, ..., -300, ..., down to "whatever 
is the lowest frequency above 0". He says he just used linspace bc he was being 
a little lazy and linspace was just faster. Weird that he wouldn't just plot it
on the correct axis because now I'm confused and his explanation of what it 
should be and how it works doesn't make sense to me. At first I thought I 
understood, but I don't see how it could go up to 500 and continue onto -400.
So it increases up to 500 and then the next number is -499? Is that really how
it get returns? 

Why is it not a symmetric result? 
Symmetry does typically always happens when you take the fourier transform of a 
real signal. The signal we transform was a complex morlet wavelt. It has 
complex parts. What's even more interesting is that the 'negative' parts of the
x-axis are empty. That is a special property of complex morlet wavelet where 
"the imaginary part has sine components in the postive part of the spectrum and
sine components in the negative part of the spectrum that cancel out". Not the
best explanation but I'll continue. It is important to know that this doesn't
apply to all complex signals, just CMWs. Other complex signals will probably 
have assymetric results.
'''

plt.plot(hz, cmw_x,'k',linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Complex Morlet wavelet in the frequency domain')
plt.show()

# 3D Plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(t, cmw.real, cmw.imag)
plt.show()