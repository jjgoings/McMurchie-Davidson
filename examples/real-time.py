import numpy as np
from mmd.molecule import Molecule 
from mmd.realtime import RealTime
from mmd.utils.spectrum import genSpectra

hydrogen = """
0 1
H  0.0 0.0 0.0  
H  0.0 0.0 0.74 
"""

# init molecule and build integrals
mol = Molecule(geometry=hydrogen,basis='3-21G')

# do the SCF
mol.RHF()

# define the applied field envelope as a function of time
# here, is is a narrow gaussian envelope centered at t = 0.
def gaussian(t):
    return np.exp(-50*(t**2))

# create realtime object, setting parameters and pulse envelopes
rt = RealTime(mol,numsteps=1000,stepsize=0.05,field=0.0001,pulse=gaussian)

# propagate with Magnus2
rt.Magnus2(direction='z')
m2 = rt.dipole

# propagate with Magnus4
rt.Magnus4(direction='z')
m4 = rt.dipole

try:
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.plot(rt.time,m2,label='Magnus2',color='tab:blue')
    ax1.plot(rt.time,m4,label='Magnus4',color='tab:green')
    ax1.set_ylabel('z-dipole / Debye',color='tab:blue')
    ax1.tick_params(axis='y',labelcolor='tab:blue')
    ax1.set_xlabel('Time / au')
    ax1.legend(loc=1)
    # plot field on separate axis
    ax2 = ax1.twinx()
    ax2.plot(rt.time,np.asarray(rt.shape)*rt.field,label='Applied field',color='tab:orange')
    ax2.set_ylabel('Applied field / au',color='tab:orange')
    ax2.tick_params(axis='y',labelcolor='tab:orange')
    ax2.legend(loc=2)

    fig.tight_layout()
    plt.show()
    plt.close()

    # now plot the absorption spectra S(w) (z-component)
    freq, spectra = genSpectra(rt.time,m2,np.asarray(rt.shape)*rt.field)
    plt.plot(27.2114*freq,spectra)
    plt.xlabel('Energy / eV')
    plt.ylabel('$\sigma(\omega)$ / arb. units')
    plt.show()

except ImportError:
    print('You need matplotlib to plot the time-evolving dipole')





