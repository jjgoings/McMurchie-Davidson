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

class test_direct(unittest.TestCase):
    def test_hydrogen_sto3g(self):
        mol = Molecule(geometry=hydrogen,basis='sto-3g')
        mol.RHF(direct=True)
        self.assertAlmostEqual(mol.energy.real,-1.11675930751)
    def test_water_sto3g(self):
        mol = Molecule(geometry=water,basis='sto-3g')
        mol.RHF(direct=True)
        self.assertAlmostEqual(mol.energy.real,-74.942079928192)
    def test_methane_sto3g(self):
        mol = Molecule(geometry=methane,basis='sto-3g')
        mol.RHF(direct=True)
        self.assertAlmostEqual(mol.energy.real,-39.726850324347)
    def test_water_DZ(self):
        mol = Molecule(geometry=water,basis='DZ')
        mol.RHF(direct=True)
        self.assertAlmostEqual(mol.energy.real,-75.977878975377)

