from mmd.molecule import * 
from mmd.spectra import *
import matplotlib.pyplot as plt

# if we do load, we will load all important integrals
load = False 

geometry = '/Users/jjgoings/Dropbox/Code/mcmurchie-davidson/geoms/h2o2.dat'

mol = Molecule(filename=geometry,basis='sto-3g',load=load,gauge=[0.0,1.0,2.0])

mol.SCF()

N  = 5000
dt = 0.05

mol.RT(direction='x',numsteps=N,stepsize=dt)
freq, xsignal = genSpectra(mol.time,mol.dipole,mol.field)
mol.RT(direction='y',numsteps=N,stepsize=dt)
freq, ysignal = genSpectra(mol.time,mol.dipole,mol.field)
mol.RT(direction='z',numsteps=N,stepsize=dt)
freq, zsignal = genSpectra(mol.time,mol.dipole,mol.field)

signal = xsignal + ysignal + zsignal
peaks(signal,freq,thresh=1.0,number=10)
plt.plot(freq*27.2114,signal)
#plt.plot(freq*27.2114,xsignal)
#plt.plot(freq*27.2114,ysignal)
#plt.plot(freq*27.2114,zsignal)
#plt.plot(np.arange(len(mol.dipole)),mol.dipole)
plt.show()






