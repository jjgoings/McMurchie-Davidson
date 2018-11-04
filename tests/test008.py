import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 

formaldehyde = """
0 1
C          0.0000000000        0.0000000000       -0.5265526741
O          0.0000000000        0.0000000000        0.6555124750
H          0.0000000000       -0.9325664988       -1.1133424527
H          0.0000000000        0.9325664988       -1.1133424527
"""

def test_formaldehyde_sto3g():
    mol = Molecule(geometry=formaldehyde,basis='sto-3g')
    mol.RHF()
    #p 6D int=acc2e=14 scf(conver=12) rhf/6-31++G** symmetry=none
    assert_allclose(mol.energy.real,-112.351590112,atol=1e-12)
    ref_dipole = np.array([0.0,0.0,-1.4821])
    assert_allclose(mol.mu,ref_dipole,atol=1e-4)

