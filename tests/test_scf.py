import unittest
import numpy as np
from mmd.integrals import *
from mmd.molecule import *
from mmd.scf import *

class test_SCF(unittest.TestCase):
    def test_water_sto3g(self):
        geom = './geoms/test/h2o.dat'
        mol = Molecule(filename=geom,basis='sto-3g')
        mol.build()
        scf = SCF(mol)
        scf.RHF()
        self.assertAlmostEqual(scf.mol.energy.real,-74.942079928192)
    def test_methane(self):
        geom = './geoms/test/ch4.dat'
        mol = Molecule(filename=geom,basis='sto-3g')
        mol.build()
        scf = SCF(mol)
        scf.RHF()
        self.assertAlmostEqual(scf.mol.energy.real,-39.726850324347)
    def test_water_DZ(self):
        geom = './geoms/test/h2o.dat'
        mol = Molecule(filename=geom,basis='DZ')
        mol.build()
        scf = SCF(mol)
        scf.RHF()
        self.assertAlmostEqual(scf.mol.energy.real,-75.977878975377)
    def test_water_321g(self):
        geom = './geoms/test/h2o.dat'
        mol = Molecule(filename=geom,basis='3-21g')
        mol.build()
        scf = SCF(mol)
        scf.RHF()
        self.assertAlmostEqual(scf.mol.energy.real,-75.5613125965)
