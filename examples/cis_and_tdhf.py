from mmd.molecule import Molecule 
from mmd.postscf import PostSCF

water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

# init molecule and build integrals
mol = Molecule(geometry=water,basis='sto-3g')

# do the SCF
mol.RHF()

# do some excited state calculations, with a few different algorithms for TDHF 
PostSCF(mol).CIS()
PostSCF(mol).TDHF()
PostSCF(mol).TDHF(alg='reduced')
PostSCF(mol).TDHF(alg='full')
