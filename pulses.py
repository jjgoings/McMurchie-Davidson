import numpy as np
from mmd.molecule import * 
from mmd.realtime import * 
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
from scipy.fftpack import hilbert
from scipy.signal import hilbert as analytic
import scipy.interpolate
from scipy.signal import correlate
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy.integrate import trapz
from sklearn import decomposition

molecule = """
0 1
H 0.0 0.0 0.74
H 0.0 0.0 0.00
"""

# init molecule and build integrals
mol = Molecule(geometry=molecule,basis='6-31ppg')
#mol = Molecule(geometry=molecule,basis='sto-3g')

# do the SCF
mol.RHF(DIIS=True,direct=False)

# Make realtime
N     = 2000
dt    = 0.10
Emax  = 0.0001

def mypulse(t):
    pulse = 0.0 + 0.0j
    pulse += np.exp(-(t)**2/(dt*2))
    pulse += np.exp(-(t-tau)**2/(dt*2))
    return pulse

taus = np.arange(5,20.0,1.0)
specs = []

# get reference signal
tau = 0.0
rt = RealTime(mol,numsteps=N+int(max(taus)/dt)+1,stepsize=dt,field=1.5*Emax,pulse=mypulse)
rt.Magnus2(direction='z')
start = int(tau/dt)
refdip = np.asarray(rt.dipole)

# Now do the pulses with delays 'tau'
cmap = mpl.cm.viridis
for i,tau in enumerate(tqdm(taus)):
    start = int(tau/dt)
    rt = RealTime(mol,numsteps=N+start,stepsize=dt,field=Emax,pulse=mypulse)
    rt.Magnus2(direction='z')
    time = rt.time[start:] - rt.time[start]
    #dipole = rt.dipole[start:]
    dipole = np.asarray(rt.dipole[start:])#-refdip[start:(N+start)]

    signal,freq = pade(time,dipole,start=0.0,stop=2.0,step=0.0005)
    field,freq = pade(time,Emax*(mypulse(time,i)+0.0001),start=0.0,stop=2.0,step=0.0005)
    signal *= freq*np.conjugate(field)/(np.dot(field,np.conjugate(field)))
    #signal = dipole 
    I_sig = trapz(np.imag(signal[200:-200]),freq[200:-200])
    hil = hilbert(signal)
    I_hil = trapz(np.imag(hil[200:-200]),freq[200:-200])
    phase = np.arctan2(I_hil,I_sig)

    #plt.plot(freq,np.imag(signal*np.exp(1j*phase)),color=cmap(i/float(len(taus))))
    plt.plot(freq,np.abs(signal*np.exp(1j*phase)),color=cmap(i/float(len(taus))))
    #plt.plot(np.real(signal),label='real')
    #plt.plot(np.imag(signal),label='imag')
    #plt.plot(freq,np.imag(hil),label='hilbert')
    #plt.plot(signal)

    specs.append(signal*np.exp(1j*phase))


S = np.vstack(specs)
#plt.show()
plt.close()

np.save('outphase.npy',S)

cov = np.cov(np.abs(specs),rowvar=False,bias=False)

#cov = np.corrcoef(np.abs(specs),rowvar=False)
#plt.imshow(np.real(S),aspect='auto',origin='lower',cmap='coolwarm')
#plt.show()
#plt.close()
#plt.contourf(cov,cmap='coolwarm')
frequency = freq*27.2114
#spread = min([abs(cov.min()),abs(cov.max())])/500
spread = min([abs(cov.min())])

#levels = np.linspace(-spread,spread,30)
#levels = np.linspace(-0.01,0.01,30)
#plt.contourf(cov,extent=[frequency.min(),frequency.max(),frequency.min(),frequency.max()],levels=levels,extend='both',cmap='coolwarm')
#plt.contourf(cov,extent=[frequency.min(),frequency.max(),frequency.min(),frequency.max()],extend='both',cmap='coolwarm')
#plt.contourf(cov,extent=[frequency.min(),frequency.max(),frequency.min(),frequency.max()],cmap='coolwarm',level=levels)
plt.contourf(cov,cmap='coolwarm')
plt.colorbar()

plt.show()

