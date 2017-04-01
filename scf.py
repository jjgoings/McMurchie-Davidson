import numpy as np
from mmd.molecule import * 
from mmd.spectra import *
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,precision=6,linewidth=200)#,formatter={'float': lambda x: format(x, '10.5E')})
# if we do load, we will load all important integrals
load = False 

#geometry = './geoms/h2o2-plus.dat'
#geometry = './geoms/h2o2.dat'
#geometry = './geoms/h2o2-far.dat'
#geometry = './geoms/h2o.dat'
#geometry = './geoms/methane.dat'
#geometry = './geoms/o2.dat'
geometry = './geoms/h4.dat'

#mol = Molecule(filename=geometry,basis='sto-3g',load=load,gauge=[10.0,10.0,10.0])
#mol = Molecule(filename=geometry,basis='sto-3g',load=load,gauge=[124000.0,0.0,0.0])
mol = Molecule(filename=geometry,basis='sto-3g',load=load,gauge=[0.0,0.0,0.0])
#mol = Molecule(filename=geometry,basis='sto-3g',load=load)

P = np.load('dens.npy')
F = np.load('fock.npy')
mol.P = np.zeros_like(mol.Core) 
#mol.P = 0.5*P.T
#print 2*np.real(mol.P)
#print 2*np.imag(mol.P)
mol.SCF()
#mol.buildFock()
#mol.computeEnergy()
#print mol.energy
#print 2*np.real(mol.F)
#print 2*np.imag(mol.F)
#print 2*np.real(mol.P)
#print 2*np.imag(mol.P)
#mol.SCF()
#print 2*np.real(mol.P)
#print 2*np.imag(mol.P)
#print mol.J
#print mol.K
#mol.Stable()
#mol.SCF()
#mol.SCF(method='imagtime')
#print "dSdB", np.real(mol.dSdB)
#print "dHdB", np.real(mol.dHdB)
#print "dGdB", np.real(mol.dGdB)
#print mol.Mx
#print mol.P
field = 0.0001
N  = 1000
dt = 0.1

mol.RT(direction='x',numsteps=N,stepsize=dt,field=field)
#x = np.asarray(mol.dipole)
freq, xsignal = genSpectra(mol.time,mol.dipole,mol.field)
freq, Lx = genSpectra(mol.time,mol.angmom,mol.field)
mol.RT(direction='y',numsteps=N,stepsize=dt,field=field)
freq, Ly = genSpectra(mol.time,mol.angmom,mol.field)
#y = np.asarray(mol.dipole)
freq, ysignal = genSpectra(mol.time,mol.dipole,mol.field)
mol.RT(direction='z',numsteps=N,stepsize=dt,field=field)
freq, Lz = genSpectra(mol.time,mol.angmom,mol.field)
#z = np.asarray(mol.dipole)
freq, zsignal = genSpectra(mol.time,mol.dipole,mol.field)
#freq, signal = genSpectra(mol.time,x+y+z,mol.field)


signal = xsignal + ysignal + zsignal
Lsignal = Lx + Ly + Lz
peaks(signal,freq,thresh=1.0,number=20)
peaks(Lsignal,freq,thresh=1.0,number=20)
#print "X"
#peaks(xsignal,freq,thresh=1.0,number=10)
#print "Y"
#peaks(ysignal,freq,thresh=1.0,number=10)
#print "Z"
#peaks(zsignal,freq,thresh=1.0,number=10)
plt.plot(freq*27.2114,signal,label='GIAO')
plt.plot(freq*27.2114,Lsignal,label='length')
#plt.plot(freq*27.2114,xsignal)
#plt.plot(freq*27.2114,ysignal)
#plt.plot(freq*27.2114,zsignal)
#mol.dipole = np.asarray(mol.dipole)[1:]
#mol.dipole -= mol.dipole[1]
#plt.plot(np.arange(len(x+y+z)),np.asarray(x+y+z))
#plt.plot(np.arange(len(x)),np.asarray(x))
#plt.plot(np.arange(len(y)),np.asarray(y))
#plt.plot(np.arange(len(z)),np.asarray(z))
#plt.plot(np.arange(len(mol.angmom)),np.asarray(mol.angmom),label='non-GIAO L')
#plt.plot(np.arange(len(mol.Hbmom)),2*np.asarray(mol.Hbmom),label='2dHdB')
#plt.plot(np.arange(len(mol.Gbmom)),np.asarray(mol.Gbmom),label='dGdB')
#plt.plot(np.arange(len(mol.Hbmom)),2*np.asarray(mol.Hbmom)+np.asarray(mol.Gbmom),label='2H + G')
#plt.plot(np.arange(len(mol.Hbmom)),2*np.asarray(mol.Hbmom)+np.asarray(mol.Gbmom)-np.asarray(mol.Sbmom),label='2H + G - S')
#plt.plot(np.arange(len(mol.Sbmom)),-2*np.asarray(mol.Sbmom),label='-2dSdB')
plt.legend()
plt.show()
