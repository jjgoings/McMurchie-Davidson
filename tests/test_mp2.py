import unittest
import numpy as np
from mmd.integrals import *
from mmd.molecule import *
from mmd.scf import *
from mmd.postscf import *

methane = """
0 1
C    0.000000    0.000000    0.000000
H    0.626425042   -0.626425042   -0.626425042
H    0.626425042    0.626425042    0.626425042
H   -0.626425042    0.626425042   -0.626425042
H   -0.626425042   -0.626425042    0.626425042
"""

class test_SCF(unittest.TestCase):
    def test_methane_sto3g(self):
        mol = Molecule(geometry=methane,basis='3-21G')
        mol.RHF(direct=False) # we need the two-electron integrals
        PostSCF(mol).MP2()
        self.assertAlmostEqual(mol.energy.real,-39.9768654272)
        self.assertAlmostEqual(mol.emp2.real,-40.076963354817)

