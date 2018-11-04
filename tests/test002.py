import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 


water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

def test_water_sto3g():
    mol = Molecule(geometry=water,basis='sto-3g')
    mol.RHF(tol=1e-14)
    assert_allclose(mol.energy.real,-74.942079928192,atol=1e-12)
    ref_dipole = np.array([0.0,1.5340,0.0])
    assert_allclose(mol.mu,ref_dipole,atol=1e-4)
#    mol.forces()
#    ref_forces = np.array([[  0.000000000,  0.097441437,  0.000000000],
#                           [ -0.086300098, -0.048720718, -0.000000000],
#                           [  0.086300098, -0.048720718, -0.000000000]])
#    assert_allclose(mol._forces,ref_forces,atol=1e-12)

