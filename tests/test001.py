import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 

hydrogen = """
0 1
H 0.0 0.0 0.74
H 0.0 0.0 0.00
"""

def test_hydrogen_sto3g():
    mol = Molecule(geometry=hydrogen,basis='sto-3g')
    mol.RHF()
    assert_allclose(mol.energy.real,-1.11675930740,atol=1e-12)
    mol.forces()
    ref_forces = np.array([[ 0.000000000, -0.000000000, -0.027679601],
                           [-0.000000000,  0.000000000,  0.027679601]])
    assert_allclose(mol._forces,ref_forces,atol=1e-12)

