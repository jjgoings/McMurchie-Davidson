import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 


methane = """
0 1
C    0.000000    0.000000    0.000000
H    0.626425042   -0.626425042   -0.626425042
H    0.626425042    0.626425042    0.626425042
H   -0.626425042    0.626425042   -0.626425042
H   -0.626425042   -0.626425042    0.626425042
"""

def test_methane_sto3g():
    mol = Molecule(geometry=methane,basis='sto-3g')
    mol.RHF(direct=True)
    assert_allclose(mol.energy.real,-39.726850324347,atol=1e-12)

