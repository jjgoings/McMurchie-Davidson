from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
from mmd.realtime import * 

hydrogen = """
0 1
H  0.0 0.0 0.0  
H  0.0 0.0 0.74 
"""

water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""


# init molecule and build integrals
mol = Molecule(geometry=hydrogen,basis='sto-3g')
mol.build()

# do the SCF
mol.RHF()

# create realtime object
rt = RealTime(mol,numsteps=50,stepsize=0.2,direction='z',field=0.001)

#
try:
   import matplotlib.pyplot as plt
   plt.plot(rt.time,rt.dipole)
   plt.show()
except ImportError:
   print('You need matplotlib to plot the time-evolving dipole')





