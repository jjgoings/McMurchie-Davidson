from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import itertools
from mmd.slater import *

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

        # TODO: Make this tx more elegant?
        # tile spin to make spin orbitals from spatial (twice dimension)
        self.mol.double_bar = np.zeros([2*idx for idx in self.mol.single_bar.shape])
        for p in range(self.mol.double_bar.shape[0]):
            for q in range(self.mol.double_bar.shape[1]):
                for r in range(self.mol.double_bar.shape[2]):
                    for s in range(self.mol.double_bar.shape[3]):
                        value1 = self.mol.single_bar[p//2,r//2,q//2,s//2].real * (p%2==r%2) * (q%2==s%2)
                        value2 = self.mol.single_bar[p//2,s//2,q//2,r//2].real * (p%2==s%2) * (q%2==r%2)
                        self.mol.double_bar[p,q,r,s] = value1 - value2

        # create Hp, the spin basis one electron operator 
        spin = np.eye(2)
        self.mol.Hp = np.kron(np.einsum('uj,vi,uv', self.mol.C, self.mol.C, self.mol.Core).real,spin)

        # create fs, the spin basis fock matrix eigenvalues 
        self.mol.fs = np.kron(np.diag(self.mol.MO),spin)


    
    def MP2(self,spin_orbital=False):
        """Routine to compute MP2 energy from RHF reference"""
        if spin_orbital:
            # Use spin orbitals from RHF reference
            EMP2 = 0.0
            occupied = range(2*self.mol.nocc)
            virtual  = range(2*self.mol.nocc,2*self.mol.nbasis)
            for i,j,a,b in itertools.product(occupied,occupied,virtual,virtual): 
                denom = self.mol.fs[i,i] + self.mol.fs[j,j] \
                      - self.mol.fs[a,a] - self.mol.fs[b,b]
                numer = self.mol.double_bar[i,j,a,b]**2 
                EMP2 += numer/denom

            self.mol.emp2 = 0.25*EMP2 + self.mol.energy   
        else:
            # Use spatial orbitals from RHF reference
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

   

    def FCI(self):
        """Routine to compute FCI energy from RHF reference"""


        print("The End")

