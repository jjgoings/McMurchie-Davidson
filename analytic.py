import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft,fftfreq
from mmd.utils.spectrum import pade

t = np.arange(0,10,0.01)

w = 2.0

ref = np.sin(w*t)

taus = np.arange(2,5,1.0)

for tau in taus:
    plt.plot(t,np.sin(w*t + tau))

plt.show()
plt.close()



