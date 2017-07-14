from __future__ import division
from __future__ import print_function
import numpy as np
from mmd.integrals import Sx, Tx, VxA, VxB, ERIx

class SCF(object):
    """SCF methods and routines for molecule object"""
    def RHF(self,doPrint=True):
        """Routine to compute the RHF energy for a closed shell molecule"""
        self.is_converged = False
        self.delta_energy = 1e20
        self.P_RMS        = 1e20
        self.P_old        = np.zeros((self.nbasis,self.nbasis)) 
        self.maxiter = 200
        if not self.is_built:
            print("You need to run Molecule.build() to generate integrals")
            self.build()
        for step in range(self.maxiter):
            if step == 0:
                self.P      = self.P_old
                self.buildFock()
                self.computeEnergy()
            else:
                self.P_old      = self.P
                energy_old = self.energy
                self.buildFock()
             
            self.orthoFock()
            E,self.CO   = np.linalg.eigh(self.FO)
    
            C      = np.dot(self.X,self.CO)
            self.C      = np.dot(self.X,self.CO)
            self.MO     = E
            self.P = np.dot(C[:,:self.nocc],np.conjugate(C[:,:self.nocc]).T)
            self.computeEnergy()
    
            if step > 0:
                self.delta_energy = self.energy - energy_old
                self.P_RMS        = np.linalg.norm(self.P - self.P_old)
            FPS = np.dot(self.F,np.dot(self.P,self.S))
            SPF = np.dot(self.S,np.dot(self.P,self.F))
            SPF = np.conjugate(FPS).T
            error = np.linalg.norm(FPS - SPF)
            if np.abs(self.P_RMS) < 1e-12 or step == (self.maxiter - 1):
                if step == (self.maxiter - 1):
                    print("NOT CONVERGED")
                else:
                    self.is_converged = True
                    FPS = np.dot(self.F,np.dot(self.P,self.S))
                    SPF = np.dot(self.S,np.dot(self.P,self.F))
                    error = FPS - SPF
                    self.computeDipole()
                    if doPrint:
                        print("E(SCF)    = ", "{0:.12f}".format(self.energy.real)+ \
                              " in "+str(step)+" iterations")
                        print("  Convergence:")
                        print("    FPS-SPF  = ", np.linalg.norm(error))
                        print("    RMS(P)   = ", "{0:.2e}".format(self.P_RMS.real))
                        print("    dE(SCF)  = ", "{0:.2e}".format(self.delta_energy.real))
                        print("  Dipole X = ", "{0:.8f}".format(self.mu[0].real))
                        print("  Dipole Y = ", "{0:.8f}".format(self.mu[1].real))
                        print("  Dipole Z = ", "{0:.8f}".format(self.mu[2].real))
                    break

    def buildFock(self):
        """Routine to build the AO basis Fock matrix"""
        self.J = np.einsum('pqrs,sr->pq', self.TwoE.astype('complex'),self.P)
        self.K = np.einsum('psqr,sr->pq', self.TwoE.astype('complex'),self.P)
        self.G = 2.*self.J - self.K
        self.F = self.Core.astype('complex') + self.G
    
    def orthoFock(self):
        """Routine to orthogonalize the AO Fock matrix to orthonormal basis"""
        self.FO = np.dot(self.X.T,np.dot(self.F,self.X))
    
    def unOrthoFock(self):
        """Routine to unorthogonalize the orthonormal Fock matrix to AO basis"""
        self.F = np.dot(self.U.T,np.dot(self.FO,self.U))
    
    def orthoDen(self):
        """Routine to orthogonalize the AO Density matrix to orthonormal basis"""
        self.PO = np.dot(self.U,np.dot(self.P,self.U.T))
    
    def unOrthoDen(self):
        """Routine to unorthogonalize the orthonormal Density matrix to AO basis"""
        self.P = np.dot(self.X,np.dot(self.PO,self.X.T))
    
    def computeEnergy(self):
        """Routine to compute the SCF energy"""
        self.el_energy = np.einsum('pq,qp',self.Core+self.F,self.P)
        self.energy    = self.el_energy + self.nuc_energy
    
    def computeDipole(self):
        """Routine to compute the SCF electronic dipole moment"""
        self.el_energy = np.einsum('pq,qp',self.Core+self.F,self.P)
        for i in range(3):
            self.mu[i] = -2*np.trace(np.dot(self.P,self.M[i])) + sum([atom.charge*(atom.origin[i]-self.gauge_origin[i]) for atom in self.atoms])  
        # to debye
        self.mu *= 2.541765
    
    def adj(self,x):
        """Returns Hermitian adjoint of a matrix"""
        assert x.shape[0] == x.shape[1]
        return np.conjugate(x).T       
    
    def comm(self,A,B):
        """Returns commutator [A,B]"""
        return np.dot(A,B) - np.dot(B,A)

    def updateFock(self):
        """Rebuilds/updates the Fock matrix if you add external fields, etc."""
        self.unOrthoDen()
        self.buildFock()
        self.orthoFock()

