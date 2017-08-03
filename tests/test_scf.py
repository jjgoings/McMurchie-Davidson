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




class test_SCF(unittest.TestCase):
    def test_hydrogen_sto3g(self):
        mol = Molecule(geometry=hydrogen,basis='sto-3g')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real,-1.11675930751)
    def test_water_sto3g(self):
        mol = Molecule(geometry=water,basis='sto-3g')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real,-74.942079928192)
        self.assertAlmostEqual(mol.mu[0].real,0.0,places=4)
        self.assertAlmostEqual(mol.mu[1].real,1.5340,places=4)
        self.assertAlmostEqual(mol.mu[2].real,0.0,places=4)
    def test_methane_sto3g(self):
        mol = Molecule(geometry=methane,basis='sto-3g')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real,-39.726850324347)
    def test_water_DZ(self):
        mol = Molecule(geometry=water,basis='DZ')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real,-75.977878975377)
        self.assertAlmostEqual(mol.mu[0].real,0.0,places=4)
        self.assertAlmostEqual(mol.mu[1].real,2.7222,places=4)
        self.assertAlmostEqual(mol.mu[2].real,0.0,places=4)
    def test_water_321g(self):
        mol = Molecule(geometry=water,basis='3-21g')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real,-75.5613125965)
    def test_he_ccpvtz(self):
        mol = Molecule(geometry=helium,basis='cc-pvtz')
        mol.RHF()
        self.assertAlmostEqual(mol.energy.real, -2.86115357403)
    def test_water_631ppgss(self):
        mol = Molecule(geometry=water,basis='6-31ppgss')
        mol.RHF()
        #p 6D int=acc2e=14 scf(conver=12) rhf/6-31++G** symmetry=none
        self.assertAlmostEqual(mol.energy.real,-75.9924381487)
        self.assertAlmostEqual(mol.mu[0].real,0.0,places=4)
        self.assertAlmostEqual(mol.mu[1].real,2.4046,places=4)
        self.assertAlmostEqual(mol.mu[2].real,0.0,places=4)
    def test_formaldehyde_sto3g(self):
        mol = Molecule(geometry=formaldehyde,basis='sto-3g')
        mol.RHF()
        #p 6D int=acc2e=14 scf(conver=12) rhf/6-31++G** symmetry=none
        self.assertAlmostEqual(mol.energy.real,-112.351590112)
        self.assertAlmostEqual(mol.mu[0].real,0.0,places=4)
        self.assertAlmostEqual(mol.mu[1].real,0.0,places=4)
        self.assertAlmostEqual(mol.mu[2].real,-1.4821,places=4)

