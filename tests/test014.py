import numpy as np
from numpy.testing import assert_allclose 
from mmd.molecule import Molecule 
from mmd.postscf import PostSCF

methane = """
0 1
C    0.000000    0.000000    0.000000
H    0.626425042   -0.626425042   -0.626425042
H    0.626425042    0.626425042    0.626425042
H   -0.626425042    0.626425042   -0.626425042
H   -0.626425042   -0.626425042    0.626425042
"""

def test_methane_321g():
    mol = Molecule(geometry=methane,basis='3-21G')
    mol.RHF(direct=False) # we need the two-electron integrals
    PostSCF(mol).MP2()
    assert_allclose(mol.energy.real,-39.9768654272)
    assert_allclose(mol.emp2.real,-40.076963354817)

