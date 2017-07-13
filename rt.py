from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
from mmd.realtime import * 

hydrogen = """
0 1
H  0.0 0.0 0.0  
H  0.0 0.0 0.74 
"""

# init molecule and build integrals
mol = Molecule(geometry=hydrogen,basis='sto-3g')
mol.build()

# do the SCF
mol.RHF()

rt = RealTime(mol)




