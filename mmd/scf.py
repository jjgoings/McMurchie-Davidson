from __future__ import division
import numpy as np
from scipy.linalg import eigh
from integrals import *

class SCF(object):
    def __init__(self,mol):
        self.mol = mol
        self.mol.is_converged = False
        self.delta_energy = 1e20
        self.P_RMS        = 1e20
        self.P_old        = np.zeros((self.mol.nbasis,self.mol.nbasis)) 
        self.maxiter = 200
        if not self.mol.is_built:
            print "You need to run Molecule.build() to generate integrals"
            self.mol.build()
            
    def RHF(self,doPrint=True):
        for step in xrange(self.maxiter):
            if step == 0:
                self.mol.P      = self.P_old
                self.buildFock()
                self.computeEnergy()
            else:
                self.P_old      = self.mol.P
                energy_old = self.mol.energy
                self.buildFock()
    
            #self.mol.FO     = np.dot(self.mol.X.T,np.dot(self.mol.F,self.mol.X))
            self.orthoFock()
            E,self.mol.CO   = np.linalg.eigh(self.mol.FO)
    
            C      = np.dot(self.mol.X,self.mol.CO)
            self.mol.C      = np.dot(self.mol.X,self.mol.CO)
            self.mol.MO     = E
            self.mol.P = np.dot(C[:,:self.mol.nocc],np.conjugate(C[:,:self.mol.nocc]).T)
            self.computeEnergy()
    
            if step > 0:
                self.delta_energy = self.mol.energy - energy_old
                self.P_RMS        = np.linalg.norm(self.mol.P - self.P_old)
            FPS = np.dot(self.mol.F,np.dot(self.mol.P,self.mol.S))
            SPF = np.dot(self.mol.S,np.dot(self.mol.P,self.mol.F))
            SPF = np.conjugate(FPS).T
            error = np.linalg.norm(FPS - SPF)
            if np.abs(self.P_RMS) < 1e-12 or step == (self.maxiter - 1):
                if step == (self.maxiter - 1):
                    print "NOT CONVERGED"
                else:
                    self.mol.is_converged = True
                    FPS = np.dot(self.mol.F,np.dot(self.mol.P,self.mol.S))
                    SPF = np.dot(self.mol.S,np.dot(self.mol.P,self.mol.F))
                    error = FPS - SPF
                    self.computeDipole()
                    if doPrint:
                        print "Error", np.linalg.norm(error)
                        print "E(SCF)    = ", "{0:.12f}".format(self.mol.energy.real)+ \
                              " in "+str(step)+" iterations"
                        print " RMS(P)  = ", "{0:.2e}".format(self.P_RMS.real)
                        print " dE(SCF) = ", "{0:.2e}".format(self.delta_energy.real)
                        print " Dipole X = ", "{0:.8f}".format(self.mol.mu_x)
                        print " Dipole Y = ", "{0:.8f}".format(self.mol.mu_y)
                        print " Dipole Z = ", "{0:.8f}".format(self.mol.mu_z)
                    break

    def buildFock(self):
        self.mol.J = np.einsum('pqrs,sr->pq', self.mol.TwoE.astype('complex'),self.mol.P)
        self.mol.K = np.einsum('psqr,sr->pq', self.mol.TwoE.astype('complex'),self.mol.P)
        self.mol.G = 2.*self.mol.J - self.mol.K
        self.mol.F = self.mol.Core.astype('complex') + self.mol.G
    
    def orthoFock(self):
        self.mol.FO = np.dot(self.mol.X.T,np.dot(self.mol.F,self.mol.X))
    
    def unOrthoFock(self):
        self.mol.F = np.dot(self.mol.U.T,np.dot(self.mol.FO,self.mol.U))
    
    def orthoDen(self):
        self.mol.PO = np.dot(self.mol.U,np.dot(self.mol.P,self.mol.U.T))
    
    def unOrthoDen(self):
        self.mol.P = np.dot(self.mol.X,np.dot(self.mol.PO,self.mol.X.T))
    
    def computeEnergy(self):
        #self.mol.el_energy = np.einsum('pq,pq',self.mol.P.T,self.mol.Core+self.mol.F)
        self.mol.el_energy = np.einsum('pq,qp',self.mol.Core+self.mol.F,self.mol.P)
        self.mol.energy    = self.mol.el_energy + self.mol.nuc_energy
    
    def computeDipole(self):
        self.mol.mu_x = -2*np.trace(np.dot(self.mol.P,self.mol.Mx)) + sum([atom.charge*(atom.origin[0]-self.mol.gauge_origin[0]) for atom in self.mol.atoms])  
        self.mol.mu_y = -2*np.trace(np.dot(self.mol.P,self.mol.My)) + sum([atom.charge*(atom.origin[1]-self.mol.gauge_origin[1]) for atom in self.mol.atoms])  
        self.mol.mu_z = -2*np.trace(np.dot(self.mol.P,self.mol.Mz)) + sum([atom.charge*(atom.origin[2]-self.mol.gauge_origin[2]) for atom in self.mol.atoms])  
        # to debye
        self.mol.mu_x *= 2.541765
        self.mol.mu_y *= 2.541765
        self.mol.mu_z *= 2.541765
    
    def adj(self,x):
        return np.conjugate(x).T       
    
    def comm(self,A,B):
        return np.dot(A,B) - np.dot(B,A)

    def forces(self):
        # compute nuclear energy gradient using finite differences. We use a 
        # simple two-point stencil. Any higher will get expensive. I'll try and
        # get analytic derivatives working in the future.
        # We do:  [f(x + h) - f(x)] / h

        if not self.mol.is_converged:
            self.exit('Need to converge SCF before computing gradient')

        # get the 3N forces on the molecule
        for atom in self.mol.atoms:
            atom.forces = []
            for direction in xrange(3):
                # form f(x + h)
                dSx = np.zeros_like(self.mol.S)
                dTx = np.zeros_like(self.mol.T)
                dVx = np.zeros_like(self.mol.V)
                dTwoEx = np.zeros_like(self.mol.TwoE)
                dVNx = 0.0
                # do one electron nuclear derivatives 
                for i in (range(self.mol.nbasis)):
                    for j in range(i+1):
                        # dSij/dx = < d phi_i/ dx | phi_j > + < phi_i | d phi_j / dx > 
                        # atom.mask is 1 if the AO involves the nuclei being differentiated, is 0 if not.
                        dSx[i,j] = dSx[j,i] \
                             = atom.mask[i]*Sx(self.mol.bfs[i],self.mol.bfs[j],n=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='A') \
                             + atom.mask[j]*Sx(self.mol.bfs[i],self.mol.bfs[j],n=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='B')
                        # dTij/dx is same form as differentiated overlaps, since Del^2 does not depend on nuclear origin 
                        dTx[i,j] = dTx[j,i] \
                             = atom.mask[i]*Tx(self.mol.bfs[i],self.mol.bfs[j],n=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='A') \
                             + atom.mask[j]*Tx(self.mol.bfs[i],self.mol.bfs[j],n=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='B')
                        # Hellman-feynman term: dVij /dx = < phi_i | d (1/r_c) / dx | phi_j >
                        dVx[i,j] = dVx[j,i] = -atom.charge*VxA(self.mol.bfs[i],self.mol.bfs[j],atom.origin,x=direction)
                        # Terms from deriv of overlap, just like dS/dx and dT/dx
                        for atomic_center in self.mol.atoms:
                            dVx[i,j] -= atom.mask[i]*atomic_center.charge*VxB(self.mol.bfs[i],self.mol.bfs[j],atomic_center.origin,x=direction,center='A')
                            dVx[i,j] -= atom.mask[j]*atomic_center.charge*VxB(self.mol.bfs[i],self.mol.bfs[j],atomic_center.origin,x=direction,center='B')
                        dVx[j,i] = dVx[i,j]

                # do nuclear repulsion contibution
                for atomic_center in self.mol.atoms:
                    # put in A != B conditions
                    RAB = np.linalg.norm(atom.origin - atomic_center.origin)
                    XAB = atom.origin[direction] - atomic_center.origin[direction]
                    ZA  = atom.charge
                    ZB  = atomic_center.charge
                    if not np.allclose(RAB,0.0):
                        dVNx += -XAB*ZA*ZB/(RAB*RAB*RAB)
                     
                # now do two electron contributions
                val = 0.0
                for i in (range(self.mol.nbasis)):
                    for j in range(i+1):
                        ij = (i*(i+1)//2 + j)
                        for k in range(self.mol.nbasis):
                            for l in range(k+1):
                                kl = (k*(k+1)//2 + l)
                                if ij >= kl:
                                   # do the four terms for gradient two electron 
                                   val = atom.mask[i]*ERIx(self.mol.bfs[i],self.mol.bfs[j],self.mol.bfs[k],self.mol.bfs[l],n1=(0,0,0),n2=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='A')
                                   val += atom.mask[j]*ERIx(self.mol.bfs[i],self.mol.bfs[j],self.mol.bfs[k],self.mol.bfs[l],n1=(0,0,0),n2=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='B')
                                   val += atom.mask[k]*ERIx(self.mol.bfs[i],self.mol.bfs[j],self.mol.bfs[k],self.mol.bfs[l],n1=(0,0,0),n2=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='C')
                                   val += atom.mask[l]*ERIx(self.mol.bfs[i],self.mol.bfs[j],self.mol.bfs[k],self.mol.bfs[l],n1=(0,0,0),n2=(0,0,0),gOrigin=self.mol.gauge_origin,x=direction,center='D')

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
                Jx = np.einsum('pqrs,sr->pq', dTwoEx, self.mol.P)
                Kx = np.einsum('psqr,sr->pq', dTwoEx, self.mol.P)
                Gx = 2.*Jx - Kx
                Fx = Hx + Gx
                force = np.einsum('pq,qp',self.mol.P,Fx + Hx) 
                # energy-weighted density matrix for overlap derivative
                W = np.dot(self.mol.P,np.dot(self.mol.F,self.mol.P)) 
                force -= 2*np.einsum('pq,qp',dSx,W)
                # nuclear-nuclear repulsion contribution
                force += dVNx
                # save forces (not mass weighted) and reset geometry
                atom.forces.append(np.real(force))



 

                
 





    
