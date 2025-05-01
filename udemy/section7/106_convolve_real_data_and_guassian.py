import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat("./ANTS_matlab/v1_laminar.mat")

srate = data['srate']
timevec = data['timevec'].T

# Signal will be ERP data from channel 7
signal = np.mean(data['csd'][6, :, :], axis=1)

# Create a Gaussian and normalize it
h = .05; # FWHM (seconds)
gtime =  np.arange(-1,1,1/srate)
gaus = np.exp( -4*np.log(2)*gtime**2 / h**2 )
gaus = gaus/sum(gaus) 

# Step 1: N's of Convolution
ndata = len(signal)
nkern = len(gaus)
nConv = ndata + nkern - 1 # convolution length
halfK = nkern // 2

# Step 2: FFT the Signal and Kernel
dataX = fft(signal, n=nConv)
kernX = fft(gaus, n=nConv)

# plt.plot(dataX)
# plt.plot(kernX)
# plt.show()

# Step 3: Multiply the Spectra
convresX = dataX * kernX

# Step 4: IFFT the Spectrum back to Time Domain
convres = ifft(convresX)

# Step 5: Cut off the "Wings"
convres = convres[halfK : -halfK + 1]
plt.plot(timevec, signal, label='Original ERP')
plt.plot(timevec, convres.real,'r',linewidth=2, label='Gaussian-convolved')

plt.xlim([-.1, 1.4])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Activity (\muV)')
plt.show()