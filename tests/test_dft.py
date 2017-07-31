import unittest
import numpy as np
from mmd.integrals import *
from mmd.molecule import *
from mmd.scf import *

water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

methane = """
0 1
C    0.000000    0.000000    0.000000
H    0.626425042   -0.626425042   -0.626425042
H    0.626425042    0.626425042    0.626425042
H   -0.626425042    0.626425042   -0.626425042
H   -0.626425042   -0.626425042    0.626425042
"""

helium = """
0 1
He 0.0 0.0 0.0
"""

hydrogen = """
0 1
H 0.0 0.0 0.74
H 0.0 0.0 0.00
"""


class test_DFT(unittest.TestCase):
    def test_helium_321g_svwn5(self):
        mol = Molecule(geometry=helium,basis='3-21G')
        mol.build()
        mol.RHF(DFT='lda')
        self.assertAlmostEqual(mol.energy.real,-2.806017308)


