from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
import numpy as np

np.set_printoptions(suppress=True,precision=6)

# read in geometry
geometry = './geoms/h2o.dat'

# init molecule and build integrals
mol = Molecule(filename=geometry,basis='sto-3g',gauge=[0.0,0.0,0.0])
mol.build()

# do the SCF
scf = SCF(mol)
scf.RHF()


# do MP2
#mp2 = PostSCF(mol)
#mp2.MP2()




