import unittest
from mmd.integrals import *
from mmd.molecule import *

aOrigin = [0.0, 0.0, 0.0]
aShell = (0,0,0) # p-orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
aExps = [3.42525091, 0.62391373, 0.16885540]
aCoefs = [0.15432897, 0.53532814, 0.44463454]
a = BasisFunction(origin=aOrigin,shell=aShell,exps=aExps,coefs=aCoefs)

class test_integrals(unittest.TestCase):
    a = BasisFunction(origin=aOrigin,shell=aShell,exps=aExps,coefs=aCoefs)
    def test_S(self):
        self.assertAlmostEqual(S(a,a),1)
        
        
