from mmd.molecule import Molecule 
from mmd.realtime import RealTime

hydrogen = """
0 1
H  0.0 0.0 0.0  
H  0.0 0.0 0.74 
"""

# init molecule and build integrals
mol = Molecule(geometry=hydrogen,basis='sto-3g')

# do the SCF
mol.RHF()

# define the applied field envelope as a function of time
# here, is is a gaussian entered at t = 0.
def envelope(t):
    gaussian = np.exp(-(t**2)) 
    return gaussian

# create realtime object, setting parameters and pulse envelopes
rt = RealTime(mol,numsteps=100,stepsize=0.2,field=0.0001,pulse=None)

# propagate with Magnus2
rt.Magnus2(direction='z')
m2 = rt.dipole
# propagate with Magnus4
rt.Magnus4(direction='z')
m4 = rt.dipole

try:
    import matplotlib.pyplot as plt
    plt.plot(rt.time,m2,label='Magnus2')
    plt.plot(rt.time,m4,label='Magnus4')
    plt.plot(rt.time,np.asarray(rt.shape)*rt.field,label='Applied field')
    plt.legend()
    plt.show()
except ImportError:
    print('You need matplotlib to plot the time-evolving dipole')





