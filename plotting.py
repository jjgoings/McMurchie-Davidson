import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from mmd.molecule import * 
from mmd.realtime import * 
from mpl_toolkits.mplot3d import Axes3D
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import scipy.interpolate
from matplotlib import cm
import matplotlib.gridspec as gridspec
import sys
from scipy.integrate import trapz
from scipy.fftpack import hilbert


data = np.load('timedata.npy')
print(data.shape)


#data = np.vsplit(data,10)[0]
#print data.shape

num_tdel,num_tsig = data.shape
tdel = np.arange(0,num_tdel)*0.25 # delay times
tsig = np.arange(0,num_tsig)*0.1 # measured signal length



plt.contourf(tsig*0.024,tdel*0.024,data)
plt.xlabel('Signal time / fs')
plt.ylabel('Delay time / fs')
plt.show()
plt.close()

#for i in np.arange(0,20):
#    #plt.plot(tdel*0.024,data[:,num_tsig//2 + int(i)])
#    plt.plot(tsig*0.024,data[i,:])
#plt.show()
#plt.close()


# Fourier transform over signal first 
Swt = []
for i in range(num_tdel):
    sig,freqsig = pade(tsig,data[i,:],start=0.2,stop=0.9,step=0.0005)
    # rephase so that absorptive peaks remain absorptive
    # note that hilbert TX is ill behaved at edges, hence the slicing
    I_sig = trapz(np.imag(sig[200:-200]),freqsig[200:-200])
    hil = hilbert(sig)
    I_hil = trapz(np.imag(hil[200:-200]),freqsig[200:-200])
    phase = np.arctan2(I_hil,I_sig)
    Swt.append(freqsig*sig*np.exp(1j*phase))


Swt = np.vstack(Swt)
print(Swt.shape)
num_tdel, num_w1 = Swt.shape

plt.contourf(tdel*0.024,freqsig*27.2114,np.imag(Swt).T)
plt.xlabel('Delay time / fs')
plt.ylabel('$\omega_1$ / eV')
plt.show()
plt.close()

np.save('tsig.npy',tsig)
np.save('swt.npy',Swt)


# Fourier transform over signal length
Sww = []
for w in range(num_w1):
    sig,freqdel = pade(tdel,Swt[:,w],start=0.2,stop=0.9,step=0.0005)
    #I_sig = trapz(np.imag(sig[200:-200]),freqdel[200:-200])
    #hil = hilbert(sig)
    #I_hil = trapz(np.imag(hil[200:-200]),freqdel[200:-200])
    #phase = np.arctan2(I_hil,I_sig)
    phase = 0.0
    Sww.append(freqdel*sig*np.exp(1j*phase))

Sww = np.vstack(Sww)
print(Sww.shape)

vmin = np.real(Sww.min())
vmax = np.real(Sww.max())


plt.contourf(freqdel*27.2114,freqsig*27.2114,np.real(Sww),30,cmap=cm.coolwarm,extend='both')
plt.title('Real')
plt.xlabel('$\omega_1$ / eV')
plt.ylabel('$\omega_3$ / eV')
plt.show()
plt.close()
#
plt.contourf(freqdel*27.2114,freqsig*27.2114,np.imag(Sww),30,cmap=cm.coolwarm,extend='both')
plt.title('Imag')
plt.xlabel('$\omega_1$')
plt.ylabel('$\omega_3$')
plt.show()
plt.close()

plt.contourf(freqdel*27.2114,freqsig*27.2114,np.abs(Sww),30,cmap=cm.coolwarm,extend='both')
plt.title('Abs')
plt.xlabel('$\omega_1$')
plt.ylabel('$\omega_3$')
plt.show()
plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(freqdel,freqsig)
surf = ax.plot_surface(X*27.2114,Y*27.2114,np.imag(Sww),cmap=cm.coolwarm)
plt.show()
