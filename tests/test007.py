import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 


helium = """
0 1
He    0.000000    0.000000    0.000000
"""

def test_helium_ccpvtz():
    mol = Molecule(geometry=helium,basis='cc-pvtz')
    mol.RHF()
    assert_allclose(mol.energy.real,-2.86115357403,atol=1e-12)

