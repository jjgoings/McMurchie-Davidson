import unittest
import numpy as np
from mmd.molecule import Molecule
from mmd.realtime import RealTime

hydrogen = """
0 1
H 0.0 0.0 0.74
H 0.0 0.0 0.00
"""

class test_RT(unittest.TestCase):
    def test_Mag2_trivial(self):
        mol = Molecule(geometry=hydrogen,basis='sto-3g')
        rt = RealTime(mol,numsteps=100,stepsize=0.2,field=0.0001,pulse=None)
        initial_energy = rt.mol.energy.real
        rt.Magnus2(direction='z')
        np.testing.assert_allclose(initial_energy*np.ones_like(rt.Energy),
                                   rt.Energy)
    def test_Mag4_trivial(self):
        mol = Molecule(geometry=hydrogen,basis='sto-3g')
        rt = RealTime(mol,numsteps=100,stepsize=0.2,field=0.0001,pulse=None)
        initial_energy = rt.mol.energy.real
        rt.Magnus4(direction='z')
        np.testing.assert_allclose(initial_energy*np.ones_like(rt.Energy),
                                   rt.Energy)


