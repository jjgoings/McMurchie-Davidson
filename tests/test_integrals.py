import unittest
import numpy as np
from mmd.integrals import *
from mmd.molecule import *
from mmd.scf import *

# From Carbon 6-31G**
# S exponents 
Se = [3047.5249000,457.3695100,103.9486900,29.2101550,9.2866630,3.1639270] 
# S coefficients 
Sc = [0.0018347,0.0140373,0.0688426,0.2321844,0.4679413,0.3623120]        
# P exponents 
Pe = [7.8682724,1.8812885,0.5442493]  
# P coefficients 
Pc = [0.0689991,0.3164240,0.7443083]        
# D exponents 
De = [0.8000000] 
# D coefficients 
Dc = [1.0000000]

# Atom 1 
s1   = BasisFunction([0,0,0],(0,0,0),Se,Sc)
p1x  = BasisFunction([0,0,0],(1,0,0),Pe,Pc)
d1xx = BasisFunction([0,0,0],(2,0,0),De,Dc)

# Atom 2
s2   = BasisFunction([2,0,0],(0,0,0),Se,Sc)
p2x  = BasisFunction([2,0,0],(1,0,0),Pe,Pc)
d2xx = BasisFunction([2,0,0],(2,0,0),De,Dc)

class test_integrals(unittest.TestCase):
    def test_S(self):
        # self overlaps
        self.assertAlmostEqual(S(s2,s2),1)
        self.assertAlmostEqual(S(p2x,p2x),1)
        self.assertAlmostEqual(S(d2xx,d2xx),1)
        # overlaps with S
        self.assertAlmostEqual(S(s1,s2),0.000257231)
        self.assertAlmostEqual(S(s1,p2x),-0.115018581165)
        self.assertAlmostEqual(S(s1,d2xx),0.177080256092)
        self.assertAlmostEqual(S(p1x,p2x),-0.3739122083249)
        self.assertAlmostEqual(S(p1x,d2xx),0.45586902867)
        self.assertAlmostEqual(S(d1xx,d2xx),0.46032406102)
    def test_T(self):
        # self kinetic 
        self.assertAlmostEqual(T(s2,s2),16.2075631760)
        self.assertAlmostEqual(T(p2x,p2x),2.178332480866)
        self.assertAlmostEqual(T(d2xx,d2xx),1.7333333333333327)
        # kinetic off diagonal 
        self.assertAlmostEqual(T(s1,s2),-0.004453163473)
        self.assertAlmostEqual(T(s1,p2x),-0.02917787713063)
        self.assertAlmostEqual(T(s1,d2xx),0.122632029183)
        self.assertAlmostEqual(T(p1x,p2x),-0.6425660209155)
        self.assertAlmostEqual(T(p1x,d2xx),0.803805220928)
        self.assertAlmostEqual(T(d1xx,d2xx),0.5532502983768)
        
        
        
        
