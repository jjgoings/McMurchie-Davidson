from __future__ import division
from __future__ import print_function
import numpy as np
from mmd.integrals.grad import Sx, Tx, VxA, VxB, ERIx

class Forces(object):
    """Nuclear gradient methods and routines for molecule object"""
    def forces(self):
        """Compute the nuclear forces"""

        if not self.is_converged:
            self.exit('Need to converge SCF before computing gradient')

        # get the 3N forces on the molecule
        for atom in self.atoms:
            # reset forces to zero
            atom.forces = np.zeros(3)
            for direction in range(3):
                # init derivative arrays
                dSx = np.zeros_like(self.S)
                dTx = np.zeros_like(self.T)
                dVx = np.zeros_like(self.V)
                dTwoEx = np.zeros_like(self.TwoE)
                dVNx = 0.0
                # do one electron nuclear derivatives 
                for i in (range(self.nbasis)):
                    for j in range(i+1):
                        # dSij/dx = 
                        #   < d phi_i/ dx | phi_j > + < phi_i | d phi_j / dx > 
                        # atom.mask is 1 if the AO involves the nuclei being 
                        # differentiated, is 0 if not.
                        dSx[i,j] = dSx[j,i] \
                             = atom.mask[i]*Sx(self.bfs[i],self.bfs[j],
                                               x=direction,center='A') \
                             + atom.mask[j]*Sx(self.bfs[i],self.bfs[j],
                                               x=direction,center='B')
                        # dTij/dx is same form as differentiated overlaps, 
                        # since Del^2 does not depend on nuclear origin 
                        dTx[i,j] = dTx[j,i] \
                             = atom.mask[i]*Tx(self.bfs[i],self.bfs[j],x=direction,center='A') \
                             + atom.mask[j]*Tx(self.bfs[i],self.bfs[j],x=direction,center='B')
                        # Hellman-feynman term: dVij /dx = < phi_i | d (1/r_c) / dx | phi_j >
                        dVx[i,j] = dVx[j,i] = -atom.charge*VxA(self.bfs[i],self.bfs[j],atom.origin,x=direction)
                        # Terms from deriv of overlap, just like dS/dx and dT/dx
                        for atomic_center in self.atoms:
                            dVx[i,j] -= atom.mask[i]*atomic_center.charge*VxB(self.bfs[i],self.bfs[j],atomic_center.origin,x=direction,center='A')
                            dVx[i,j] -= atom.mask[j]*atomic_center.charge*VxB(self.bfs[i],self.bfs[j],atomic_center.origin,x=direction,center='B')
                        dVx[j,i] = dVx[i,j]

                # do nuclear repulsion contibution
                for atomic_center in self.atoms:
                    # put in A != B conditions
                    RAB = np.linalg.norm(atom.origin - atomic_center.origin)
                    XAB = atom.origin[direction] - atomic_center.origin[direction]
                    ZA  = atom.charge
                    ZB  = atomic_center.charge
                    if not np.allclose(RAB,0.0):
                        dVNx += -XAB*ZA*ZB/(RAB*RAB*RAB)
                     
                # now do two electron contributions
                val = 0.0
                for i in (range(self.nbasis)):
                    for j in range(i+1):
                        ij = (i*(i+1)//2 + j)
                        for k in range(self.nbasis):
                            for l in range(k+1):
                                kl = (k*(k+1)//2 + l)
                                if ij >= kl:
                                   # do the four terms for gradient two electron 
                                   val = atom.mask[i]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='a')
                                   val += atom.mask[j]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='b')
                                   val += atom.mask[k]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='c')
                                   val += atom.mask[l]*ERIx(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],x=direction,center='d')
                                   # we have exploited 8-fold permutaitonal symmetry here
                                   dTwoEx[i,j,k,l] = val
                                   dTwoEx[k,l,i,j] = val
                                   dTwoEx[j,i,l,k] = val
                                   dTwoEx[l,k,j,i] = val
                                   dTwoEx[j,i,k,l] = val
                                   dTwoEx[l,k,i,j] = val
                                   dTwoEx[i,j,l,k] = val
                                   dTwoEx[k,l,j,i] = val

                # Fock gradient terms
                Hx = dTx + dVx
                Jx = np.einsum('pqrs,sr->pq', dTwoEx, self.P)
                Kx = np.einsum('psqr,sr->pq', dTwoEx, self.P)
                Gx = 2.*Jx - Kx
                Fx = Hx + Gx
                force = np.einsum('pq,qp',self.P,Fx + Hx) 
                # energy-weighted density matrix for overlap derivative
                W = np.dot(self.P,np.dot(self.F,self.P)) 
                force -= 2*np.einsum('pq,qp',dSx,W)
                # nuclear-nuclear repulsion contribution
                force += dVNx
                # save forces (not mass weighted) and reset geometry
                # strictly speaking we computed dE/dX, but F = -dE/dX 
                atom.forces[direction] = np.real(-force)


