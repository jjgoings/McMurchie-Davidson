import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 


water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

def test_water_DZ():
    mol = Molecule(geometry=water,basis='DZ')
    mol.RHF(direct=True)
    assert_allclose(mol.energy.real,-75.977878975377,atol=1e-12)
    ref_dipole = np.array([0.0,2.7222,0.0])
    assert_allclose(mol.mu,ref_dipole,atol=1e-4)

