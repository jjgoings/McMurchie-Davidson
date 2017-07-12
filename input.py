from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
import numpy as np

# read in geometry
geometry = './geoms/test/h2o.dat'

# init molecule and build integrals
mol = Molecule(filename=geometry,basis='sto-3g')
mol.build()

# do the SCF
scf = SCF(mol)
scf.RHF()

# compute and print nuclear forces
scf.forces()
for atom in mol.atoms:
    print atom.forces

# do MP2
mp2 = PostSCF(mol)
mp2.MP2()




