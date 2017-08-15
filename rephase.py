import numpy as np
import matplotlib.pyplot as plt

zero = np.load('inphase.npy')
one80 = np.load('outphase.npy')

signal = zero[0] - one80[0] 
plt.plot(np.imag(zero)[0])
#plt.plot(np.imag(one80)[0])
#plt.plot(np.real(zero)[0])
plt.plot(np.real(one80)[0])
#plt.plot(np.imag(signal))
plt.show()

