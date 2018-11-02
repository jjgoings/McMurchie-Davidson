import unittest
import numpy as np
from mmd.integrals.reference import S as oldS
from mmd.integrals.reference import T as oldT
from mmd.integrals.reference import V as oldV
from mmd.integrals.reference import ERI as oldERI
from mmd.integrals.onee import S,T,V
from mmd.integrals.twoe import ERI 
from mmd.integrals.twoe import Basis as BasisFunction 
from mmd.integrals.reference import BasisFunction as oldBasisFunction 

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
s1   = BasisFunction([0,0,0],(0,0,0),len(Se),Se,Sc)
p1x  = BasisFunction([0,0,0],(1,0,0),len(Pe),Pe,Pc)
d1xx = BasisFunction([0,0,0],(2,0,0),len(De),De,Dc)
# Atom 1 pure python
olds1   = oldBasisFunction([0,0,0],(0,0,0),len(Se),Se,Sc)
oldp1x  = oldBasisFunction([0,0,0],(1,0,0),len(Pe),Pe,Pc)
oldd1xx = oldBasisFunction([0,0,0],(2,0,0),len(De),De,Dc)

# Atom 2
s2   = BasisFunction([2,0,0],(0,0,0),len(Se),Se,Sc)
p2x  = BasisFunction([2,0,0],(1,0,0),len(Pe),Pe,Pc)
d2xx = BasisFunction([2,0,0],(2,0,0),len(De),De,Dc)
# Atom 2 pure python
olds2   = oldBasisFunction([2,0,0],(0,0,0),len(Se),Se,Sc)
oldp2x  = oldBasisFunction([2,0,0],(1,0,0),len(Pe),Pe,Pc)
oldd2xx = oldBasisFunction([2,0,0],(2,0,0),len(De),De,Dc)

class test_integrals(unittest.TestCase):
    def test_S(self):
        # self overlaps
        self.assertAlmostEqual(S(s2,s2),1)
        self.assertAlmostEqual(S(p2x,p2x),1)
        self.assertAlmostEqual(S(d2xx,d2xx),1)
        # python matches cython?
        self.assertAlmostEqual(S(s2,s2),oldS(s2,s2))
        self.assertAlmostEqual(S(p2x,p2x),oldS(p2x,p2x))
        self.assertAlmostEqual(S(d2xx,d2xx),oldS(p2x,p2x))

        # overlaps with S
        self.assertAlmostEqual(S(s1,s2),0.000257231)
        self.assertAlmostEqual(S(s1,p2x),-0.115018581165)
        self.assertAlmostEqual(S(s1,d2xx),0.177080256092)
        self.assertAlmostEqual(S(p1x,p2x),-0.3739122083249)
        self.assertAlmostEqual(S(p1x,d2xx),0.45586902867)
        self.assertAlmostEqual(S(d1xx,d2xx),0.46032406102)
        # python matches cython?
        self.assertAlmostEqual(S(s1,s2),oldS(s1,s2))
        self.assertAlmostEqual(S(s1,p2x),oldS(s1,p2x))
        self.assertAlmostEqual(S(s1,d2xx),oldS(s1,d2xx))
        self.assertAlmostEqual(S(p1x,p2x),oldS(p1x,p2x))
        self.assertAlmostEqual(S(p1x,d2xx),oldS(p1x,d2xx))
        self.assertAlmostEqual(S(d1xx,d2xx),oldS(d1xx,d2xx))

    def test_T(self):
        # self kinetic 
        self.assertAlmostEqual(T(s2,s2),16.2075631760)
        self.assertAlmostEqual(T(p2x,p2x),2.178332480866)
        self.assertAlmostEqual(T(d2xx,d2xx),1.7333333333333327)
        # python matches cython?
        self.assertAlmostEqual(T(s2,s2),oldT(s2,s2))
        self.assertAlmostEqual(T(p2x,p2x),oldT(p2x,p2x))
        self.assertAlmostEqual(T(d2xx,d2xx),oldT(d2xx,d2xx))

        # kinetic off diagonal 
        self.assertAlmostEqual(T(s1,s2),-0.004453163473)
        self.assertAlmostEqual(T(s1,p2x),-0.02917787713063)
        self.assertAlmostEqual(T(s1,d2xx),0.122632029183)
        self.assertAlmostEqual(T(p1x,p2x),-0.6425660209155)
        self.assertAlmostEqual(T(p1x,d2xx),0.803805220928)
        self.assertAlmostEqual(T(d1xx,d2xx),0.5532502983768)
        # python matches cython?
        self.assertAlmostEqual(T(s1,s2),oldT(s1,s2))
        self.assertAlmostEqual(T(s1,p2x),oldT(s1,p2x))
        self.assertAlmostEqual(T(s1,d2xx),oldT(s1,d2xx))
        self.assertAlmostEqual(T(p1x,p2x),oldT(p1x,p2x))
        self.assertAlmostEqual(T(p1x,d2xx),oldT(p1x,d2xx))
        self.assertAlmostEqual(T(d1xx,d2xx),oldT(d1xx,d2xx))

    def test_V(self):
        # random
        origin = np.array([-0.34523,0.4,2.0])
        # python matches cython?
        self.assertAlmostEqual(V(s2,s2,origin),oldV(s2,s2,origin))
        self.assertAlmostEqual(V(p2x,p2x,origin),oldV(p2x,p2x,origin))
        self.assertAlmostEqual(V(d2xx,d2xx,origin),oldV(d2xx,d2xx,origin))
        # python matches cython?
        self.assertAlmostEqual(V(s1,s2,origin),oldV(s1,s2,origin))
        self.assertAlmostEqual(V(s1,p2x,origin),oldV(s1,p2x,origin))
        self.assertAlmostEqual(V(s1,d2xx,origin),oldV(s1,d2xx,origin))
        self.assertAlmostEqual(V(p1x,p2x,origin),oldV(p1x,p2x,origin))
        self.assertAlmostEqual(V(p1x,d2xx,origin),oldV(p1x,d2xx,origin))
        self.assertAlmostEqual(V(d1xx,d2xx,origin),oldV(d1xx,d2xx,origin))

    def test_ERI(self):
        # somewhat randomly chosen consistency test
        for a in [s1,p2x,d1xx]:
            for b in [s2,p2x,d1xx]:
                for c in [s2,p1x,d2xx]:
                    for d in [s1,p1x,d2xx]:
                        self.assertAlmostEqual(ERI(a,b,c,d),oldERI(a,b,c,d))
       

    def test_BasisFunction(self):
        # testing pure python normalization is consistent 
        assert s1.num_exps == olds1.num_exps
        for exp in range(s1.num_exps):
            self.assertAlmostEqual(s1.coefs[exp],olds1.coefs[exp])
        assert p1x.num_exps == oldp1x.num_exps
        for exp in range(p1x.num_exps):
            self.assertAlmostEqual(p1x.coefs[exp],oldp1x.coefs[exp])
        assert d1xx.num_exps == oldd1xx.num_exps
        for exp in range(d1xx.num_exps):
            self.assertAlmostEqual(d1xx.coefs[exp],oldd1xx.coefs[exp])
        
        
        
        
        
