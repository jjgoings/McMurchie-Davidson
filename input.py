from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 

# read in geometry
geometry = './geoms/he.dat'

# init molecule and build integrals
mol = Molecule(filename=geometry,basis='6-31G')
mol.build()
print mol.S

# do the SCF
scf = SCF(mol)
scf.RHF()

# do MP2
mp2 = PostSCF(mol)
mp2.MP2()




