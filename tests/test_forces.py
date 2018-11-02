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

formaldehyde = """
0 1
C          0.0000000000        0.0000000000       -0.5265526741
O          0.0000000000        0.0000000000        0.6555124750
H          0.0000000000       -0.9325664988       -1.1133424527
H          0.0000000000        0.9325664988       -1.1133424527
"""




class test_Forces(unittest.TestCase):
    def test_hydrogen_sto3g(self):
        mol = Molecule(geometry=hydrogen,basis='sto-3g')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real,-1.11675930740,places=10)
        mol.forces()
        ref_forces = np.array([[ 0.000000000, -0.000000000, -0.027679601],
                               [-0.000000000,  0.000000000,  0.027679601]])
        np.testing.assert_allclose(mol._forces,ref_forces,atol=1e-12)

    def test_water_sto3g(self):
        mol = Molecule(geometry=water,basis='sto-3g')
        mol.RHF(tol=1e-14)
        self.assertAlmostEqual(mol.energy.real,-74.942079928192)
        mol.forces()
        ref_forces = np.array([[  0.000000000,  0.097441437,  0.000000000],
                               [ -0.086300098, -0.048720718, -0.000000000],
                               [  0.086300098, -0.048720718, -0.000000000]])
        np.testing.assert_allclose(mol._forces,ref_forces,atol=1e-12)

