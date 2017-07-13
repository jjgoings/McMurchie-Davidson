from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import itertools

class PostSCF(object):
    """Class for post-scf routines"""
    def __init__(self,mol):
        self.mol = mol
        if not self.mol.is_converged:
            sys.exit("SCF not converged, skipping Post-SCF")
        self.ao2mo()

    def ao2mo(self):
        """Routine to convert AO integrals to MO integrals"""
        self.mol.single_bar = np.einsum('mp,mnlz->pnlz',
                                        self.mol.C,self.mol.TwoE)
        temp = np.einsum('nq,pnlz->pqlz',
                         self.mol.C,self.mol.single_bar)
        self.mol.single_bar = np.einsum('lr,pqlz->pqrz',
                                        self.mol.C,temp)
        temp = np.einsum('zs,pqrz->pqrs',
                         self.mol.C,self.mol.single_bar)
        self.mol.single_bar = temp
    
    def MP2(self):
        """Routine to compute MP2 energy from RHF reference"""
        EMP2 = 0.0
        occupied = range(self.mol.nocc)
        virtual  = range(self.mol.nocc,self.mol.nbasis)
        for i,j,a,b in itertools.product(occupied,occupied,virtual,virtual): 
            denom = self.mol.MO[i] + self.mol.MO[j] \
                  - self.mol.MO[a] - self.mol.MO[b]
            numer = self.mol.single_bar[i,a,j,b] \
                  * (2.0*self.mol.single_bar[i,a,j,b] 
                    - self.mol.single_bar[i,b,j,a])
            EMP2 += numer/denom
        self.mol.emp2 = EMP2 + self.mol.energy   
        print('E(MP2) = ', self.mol.emp2.real) 

