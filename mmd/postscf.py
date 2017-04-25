from __future__ import division
import numpy as np
import sys

class PostSCF(object):
    def __init__(self,mol):
        self.mol = mol
        if not self.mol.is_converged:
            sys.exit("SCF not converged, skipping Post-SCF")
        self.ao2mo()

    def ao2mo(self):
        self.mol.single_bar = np.einsum('mp,mnlz->pnlz',self.mol.C,self.mol.TwoE)
        temp            = np.einsum('nq,pnlz->pqlz',self.mol.C,self.mol.single_bar)
        self.mol.single_bar = np.einsum('lr,pqlz->pqrz',self.mol.C,temp)
        temp            = np.einsum('zs,pqrz->pqrs',self.mol.C,self.mol.single_bar)
        self.mol.single_bar = temp
    
    def MP2(self):
        EMP2 = 0.0
        for i in range(self.mol.nocc):
            for j in range(self.mol.nocc):
                for a in range(self.mol.nocc,self.mol.nbasis):
                    for b in range(self.mol.nocc,self.mol.nbasis):
                        denom = self.mol.MO[i] + self.mol.MO[j] - self.mol.MO[a] - self.mol.MO[b]
                        numer = self.mol.single_bar[i,a,j,b]*(2.0*self.mol.single_bar[i,a,j,b] - self.mol.single_bar[i,b,j,a])
                        EMP2 += numer/denom
    
        print 'E(MP2) = ', EMP2 + self.mol.energy

