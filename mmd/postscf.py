from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import itertools
from bitstring import BitArray
from mmd.slater import common_index, get_excitation
from scipy.special import comb

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

        self.mol.norb = self.mol.nbasis * 2 # spin orbital

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
            occupied = range(self.mol.nelec)
            virtual  = range(self.mol.nelec,self.mol.norb)
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

    @staticmethod
    def tuple2bitstring(bit_tuple):
        ''' From tuple of occupied orbitals, return bitstring representation '''
        string = ['0']*(max(bit_tuple) + 1)
        for i in bit_tuple:
            string[i] = '1'
        string = ''.join(string[::-1])
        return BitArray(bin=string)
   

    def FCI(self):
        """Routine to compute FCI energy from RHF reference"""

        nEle = self.mol.nelec
        nOrb = self.mol.norb
        det_list = []
 
        if comb(nEle,nOrb) > 5000:
            print("Number determinants: ",comb(nEle,nOrb))
            sys.exit("FCI too expensive. Quitting.")
        
         
        # FIXME: limited to 64 orbitals at the moment 
        for occlist in itertools.combinations(range(nOrb), nEle):
            string = PostSCF.tuple2bitstring(occlist)
            det = np.array([string.uint])
            det_list.append(det)

        Nint = int(np.floor(nOrb/64) + 1)
        H = np.zeros((len(det_list),len(det_list)))
        print("Building Hamiltonian...")
        for idx,det1 in enumerate(det_list):
            for jdx,det2 in enumerate(det_list[:(idx+1)]):
               exc, degree, phase = get_excitation(det1,det2,Nint)

               if degree > 2:
                   continue
               elif degree == 2:
                   # sign * <hole1,hole2||particle1,particle2>
                   value = phase * self.mol.double_bar[exc[1,0], exc[2,0], exc[1,1], exc[2,1]] 
               elif degree == 1:
                   m = exc[1,0]
                   p = exc[1,1]
                   common = common_index(det1,det2,Nint)
                   tmp = self.mol.Hp[m,p]
                   for n in common:
                       tmp += self.mol.double_bar[m, n, p, n] 
                   value = phase * tmp
               elif degree == 0:
                   # kind of lazy to use common_index...
                   common = common_index(det1,det2,Nint)
                   tmp = 0.0
                   for m in common:
                       tmp += self.mol.Hp[m, m]
                   # also lazy
                   for m in common: 
                       for n in common:
                           tmp += 0.5*self.mol.double_bar[m,n,m,n]
                              
                   value = phase * tmp 

               H[idx,jdx] = value
               H[jdx,idx] = value

        print("Diagonalizing Hamiltonian...")
        E,C = np.linalg.eigh(H)
        self.mol.efci = E[0] + self.mol.nuc_energy
        
        print("\nFull Configuration Interaction")
        print("------------------------------")
        print("# Determinants: ",len(det_list))
        print("SCF energy: %12.8f" % self.mol.energy.real)
        print("FCI corr:   %12.8f" % (self.mol.efci - self.mol.energy.real))
        print("FCI energy: %12.8f" % self.mol.efci)



