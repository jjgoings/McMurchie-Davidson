import numpy as np
from mmd.molecule import * 
from mmd.realtime import * 
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import scipy.interpolate
import matplotlib as mpl

methane = """
0 1
C   0.000000    0.000000    0.000000
H   0.626425   -0.626425   -0.626425
H   0.626425    0.626425    0.626425
H  -0.626425    0.626425   -0.626425
H  -0.626425   -0.626425    0.626425
"""

h4 = """
0 1
 H                  0.00000000    0.73652600   0.00000000
 H                  0.00000000   -0.73652600   0.00000000
 H                  0.15384021    0.89589200   0.94470798
 H                 -0.95715200   -0.89589200   0.00015200
"""
h2 = """
0 1 
H 0.52 0.0 0.52
H 0.0 0.0 0.0
"""

water = """
0 1
O 0.000000   -0.075792    0.000000
H 0.866812    0.601436    0.000000
H -0.866812    0.601436    0.000000
"""



# init molecule and build integrals
mol = Molecule(geometry=h2,basis='6-31ppg')
mol.build()

# do the SCF
mol.RHF()

# Make realtime
N     = 2000
dt    = 0.10
Emax  = 0.0001

def mypulse(t):
    pulse = 0.0
    pulse += np.exp(-(t)**2/dt)
    pulse += np.exp(-(t-tau)**2/dt)
    return pulse

def delta(t):
    pulse = 0.0
    pulse += 2*np.exp(-(t-10)**2/dt)
    return pulse

taus = np.arange(10,18,0.4)
specs1 = []
specs2 = []

start = int(max(taus)/dt) + 1

rt = RealTime(mol,numsteps=N,stepsize=dt,field=Emax,pulse=delta)
rt.Magnus2(direction='z')
refZ,frequency = pade(rt.time[start:],rt.dipole[start:]-rt.dipole[0])    
refsignal = np.real(refZ*refZ.conjugate())

cmap = mpl.cm.viridis
for i,tau in enumerate(tqdm(taus)):
    rt = RealTime(mol,numsteps=N,stepsize=dt,field=Emax,pulse=mypulse)
    rt.Magnus2(direction='z')
    signalz,frequency = pade(rt.time[start:],rt.dipole[start:]-rt.dipole[0])    
    signal = np.real(signalz*signalz.conjugate()) 
    specs2.append((signal - refsignal)*frequency/np.linalg.norm(signal - refsignal))
    #plt.plot(frequency*27.2114,frequency*(signal-refsignal),color=cmap(i/float(len(taus)))) 
   #plt.plot(rt.time,np.array(rt.shape)*rt.field)
   #plt.plot(rt.time,np.array(rt.Energy))

S = np.vstack(specs2)
cov = np.cov(specs2,rowvar=False,bias=False)
#plt.imshow(cov,aspect='auto',origin='lower',cmap='coolwarm')

frequency *= 27.2114
spread = min([abs(cov.min()),abs(cov.max())])
levels = np.linspace(-spread,spread,20)
plt.contourf(cov,extent=[frequency.min(),frequency.max(),frequency.min(),frequency.max()],levels=levels,extend='both')
plt.colorbar()


plt.show()
