import numpy as np
from mmd.molecule import * 
from mmd.realtime import * 
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import scipy.interpolate
import matplotlib as mpl
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
    pulse += np.exp(-(t)**2/dt)
    return pulse

taus = np.arange(15,20,0.8)
specs1 = []
specs2 = []
start = int(max(taus)/dt) + 1

# Get the reference signal (e.g. both pulses at same time)
tau = 0.0
rt = RealTime(mol,numsteps=N,stepsize=dt,field=Emax,pulse=mypulse)
rt.Magnus2(direction='z')
refZ,frequency = pade(rt.time[start:],rt.dipole[start:]-rt.dipole[0])    
refsignal = np.real(refZ*refZ.conjugate())

# Do a few sample delta pulses and plot
cmap1 = mpl.cm.Vega20
cmap2 = mpl.cm.Vega20
for idx, tau in enumerate([15,17,20]):
    rt = RealTime(mol,numsteps=N,stepsize=dt,field=Emax,pulse=mypulse)
    rt.Magnus2(direction='z')
    #plt.plot(rt.time*0.0241888425,np.array(rt.shape)*rt.field,color=cmap1(idx/3.0))
    ax = plt.gca()
    plt.plot(rt.time*0.0241888425,-1*np.array(rt.dipole),color=cmap2(2*idx))
    ax.fill_between(rt.time*0.0241888425, np.array(rt.shape)*rt.field, 0, interpolate=True, color=cmap1(2*idx+1), alpha=0.8,label=str('{0:.2f}'.format(tau*0.024)))

tau = 0.0
rt = RealTime(mol,numsteps=N,stepsize=dt,field=Emax,pulse=delta)
rt.Magnus2(direction='z')
plt.plot(rt.time*0.0241888425,-1*np.array(rt.dipole),color='black',ls='--',label='Reference')
ax.fill_between(rt.time*0.0241888425, np.array(np.exp(-(rt.time)**2/dt))*rt.field, 0, interpolate=True, color='black')
plt.legend(title='Pulse delay / fs')


plt.xlim([0,0.80])
plt.xlabel('Time / fs')
plt.ylabel('Induced z-Dipole / arb. units')
plt.yticks([])
plt.title('Induced dipole as function of pulse delay, H$_2$ 6-31++G')
plt.savefig('pulses.pdf',bbox_inches='tight')
plt.close()

# Now do the pulses with delays 'tau'
cmap = mpl.cm.viridis
for i,tau in enumerate(tqdm(taus)):
    rt = RealTime(mol,numsteps=N,stepsize=dt,field=Emax,pulse=mypulse)
    rt.Magnus2(direction='z')
    signalz,frequency = pade(rt.time[start:],rt.dipole[start:]-rt.dipole[0])    
    signal = np.real(signalz*signalz.conjugate()) 
    specs2.append((signal-refsignal)*frequency/np.linalg.norm(signal - refsignal))
    plt.plot(frequency*27.2114,frequency*(signal-refsignal),color=cmap(i/float(len(taus))),label=str('{0:.2f}'.format(tau*0.024))) 
    #plt.plot(rt.time,np.array(rt.shape)*rt.field)
    #plt.plot(rt.time,np.array(rt.Energy))

plt.xlabel('Energy / eV')
plt.yticks([])
plt.legend(title='Pulse delay / fs',loc=2)
plt.title('$\Delta$ Absorption, H$_2$ 6-31++G')
plt.savefig('delta-absorption.pdf',bbox_inches='tight')
plt.close()

S = np.vstack(specs2) # collect time series into matrix
cov = np.cov(specs2,rowvar=False,bias=False) # create synchr. correlation matrix
frequency *= 27.2114
spread = min([abs(cov.min()),abs(cov.max())])*0.7
levels = np.linspace(-spread,spread,10)

fig = plt.figure(1)
fig.suptitle('Correlation plot, H$_2$ 6-31++G',size=14)
n = 6 # defines relative aspect between spectra and correlation plot
gridspec.GridSpec(n,n)

# large correlation plot
plt.subplot2grid((n,n), (1,1), colspan=(n-1),rowspan=(n-1))
plt.tick_params(axis='y', labelright='on',right='on')
plt.xlabel('Energy / eV')
plt.ylabel('Energy / eV')
ax = plt.gca() # get current axis object
ax.yaxis.set_label_position('right')

plt.contourf(cov,extent=[frequency.min(),frequency.max(),frequency.min(),frequency.max()],levels=levels,extend='both',cmap='coolwarm')
plt.plot(frequency,frequency,lw=0.8,ls=':',color='black')

# top spectrum
plt.subplot2grid((n,n),(0,1), colspan=(n-1))
plt.xticks([])
plt.yticks([])
plt.plot(frequency,refsignal,color='black')
plt.xlim([min(frequency),max(frequency)])

# left spectrum
plt.subplot2grid((n,n),(1,0), rowspan=(n-1))
plt.xticks([])
plt.yticks([])
plt.plot(-refsignal,frequency,color='black')
plt.ylim([min(frequency),max(frequency)])


fig.subplots_adjust(hspace=0,wspace=0)
#cbar = plt.colorbar()
#cbar.ax.set_yticklabels([])
#cbar.set_ticks([])
#cbar = plt.colorbar(format='%.0e')
#cbar.ax.set_ylabel('Correlation / arb. units')
fig.set_size_inches(w=5,h=5)#plt.show()
plt.savefig('correlation.pdf', bbox_inches='tight')
