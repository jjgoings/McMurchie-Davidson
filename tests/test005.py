import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 


water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

def test_water_321g():
    mol = Molecule(geometry=water,basis='3-21g')
    mol.RHF()
    assert_allclose(mol.energy.real,-75.5613125965,atol=1e-12)

