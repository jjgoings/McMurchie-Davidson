import numpy as np
from mmd.molecule import * 
from mmd.realtime import * 
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
from scipy.fftpack import hilbert
import scipy.interpolate
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
import sys

h2 = """
0 1 
H 0.52 0.0 0.52
H 0.0 0.0 0.0
"""
# init molecule and build integrals
mol = Molecule(geometry=h2,basis='6-31ppg')

# do the SCF
mol.RHF()

# Make realtime
N     = 500
dt    = 0.1
Emax  = 0.0001

def mypulse(t):
    pulse = 0.0
    pulse += np.exp(-(t)**2/dt)
    pulse += np.exp(-(t-tau)**2/dt)
    return pulse

#omegas = np.arange(10,25,0.5)
#taus = 2*np.pi*27.2114/omegas
# note tau must be integer divisible by dt
taus = np.arange(0,75,0.5)
specs = []

# get reference signal
tau = 0.0
rt = RealTime(mol,numsteps=N+int(max(taus)/dt)+1,stepsize=dt,field=Emax,pulse=mypulse)
rt.Magnus2(direction='z')
start = int(tau/dt)
refdip = np.asarray(rt.dipole)

# Now do the pulses with delays 'tau'
cmap = mpl.cm.viridis
for i,tau in enumerate(tqdm(taus)):
    start = int(tau/dt) + 1
    #mol = Molecule(geometry=h2,basis='6-31ppg')
    rt = RealTime(mol,numsteps=N+start,stepsize=dt,field=Emax,pulse=mypulse)
    rt.Magnus2(direction='z')
    specs.append(np.asarray(rt.dipole[start:])-refdip[start:(N+start)]) 
    time = rt.time[start:]
    #plt.plot(time,rt.dipole[start:])

specs = np.vstack(specs)

np.save('timedata.npy',specs)

plt.contourf(time,taus,specs)
plt.show()



