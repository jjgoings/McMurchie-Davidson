import numpy as np
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import hilbert

t = np.arange(0,100,0.1)

signal = (np.sin(5*t) + np.sin(3*t))*np.exp(-t/50)
h1 = hilbert(signal)

signal2 = (np.sin(5*t-2) + np.sin(3*t-2))*np.exp(-t/50)
h2 = hilbert(signal2)

p1 = np.unwrap(np.angle(h1))
p2 = np.unwrap(np.angle(h2))

w = (p2-p1)[len(p2)//2]

plt.plot(t,(p2-p1))
plt.show()
plt.close()

spec,freq = pade(t,signal,start=0.0,stop=10.0,step=0.01)
spec2,freq = pade(t,signal2,start=0.0,stop=10.0,step=0.01)

plt.plot(freq,np.imag(spec))
plt.plot(freq,np.imag(spec2*np.exp(1j*w)))
plt.plot(freq,np.imag(spec2))
plt.show()



