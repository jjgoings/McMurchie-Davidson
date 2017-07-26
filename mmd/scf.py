from __future__ import division
from __future__ import print_function
import numpy as np
from numpy.linalg import multi_dot as dot

class SCF(object):
    """SCF methods and routines for molecule object"""
    def RHF(self,doPrint=True,DIIS=True,DFT=None):
        """Routine to compute the RHF energy for a closed shell molecule"""
        self.is_converged = False
        self.delta_energy = 1e20
        self.P_RMS        = 1e20
        self.P_old        = np.zeros((self.nbasis,self.nbasis)) 
        self.maxiter = 200
        if not self.is_built:
            print("You need to run Molecule.build() to generate integrals")
            self.build()

        self.P      = self.P_old
        if DFT:
            from mmd.grid.grid import Grid
            self.grid = Grid(self.atoms)
            self.xcname = str(DFT)
            self.grid.setbfamps(self.bfs)
        else:
            self.xcname = None

        self.buildFock()

        if DIIS:
            fockSet = []
            errorSet = []

        for step in range(self.maxiter):
            if step > 0:
                self.P_old      = self.P
                energy_old = self.energy
                self.buildFock()

                if DIIS:
                    FPS =   dot([self.F,self.P,self.S])
                    SPF =   self.adj(FPS) 
                    # error must be in orthonormal basis
                    error = dot([self.X,FPS-SPF,self.X]) 
                    fockSet.append(self.F)
                    errorSet.append(error) 
                    numFock = len(fockSet)
                    # limit subspace, hardcoded for now
                    if numFock > 8:
                        del fockSet[0] 
                        del errorSet[0] 
                        numFock -= 1
                    B = np.zeros((numFock + 1,numFock + 1)) 
                    B[-1,:] = B[:,-1] = -1.0
                    B[-1,-1] = 0.0
                    # B is symmetric
                    for i in range(numFock):
                        for j in range(i+1):
                            B[i,j] = B[j,i] = \
                                np.real(np.trace(np.dot(self.adj(errorSet[i]),
                                                                 errorSet[j])))
                    residual = np.zeros(numFock + 1)
                    residual[-1] = -1.0
                    weights = np.linalg.solve(B,residual)

                    # weights is 1 x numFock + 1, but first numFock values
                    # should sum to one if we are doing DIIS correctly
                    assert np.isclose(sum(weights[:-1]),1.0)
 
                    F = np.zeros_like(self.F)
                    for i, Fock in enumerate(fockSet):
                        F += weights[i] * Fock
                    self.F = F 
             
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
            #SPF = self.adj(FPS)
            error = np.linalg.norm(FPS - SPF)
            if np.abs(self.P_RMS) < 1e-10 or step == (self.maxiter - 1):
                if step == (self.maxiter - 1):
                    print("NOT CONVERGED")
                else:
                    self.is_converged = True
                    FPS = np.dot(self.F,np.dot(self.P,self.S))
                    SPF = self.adj(FPS) 
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
        if self.xcname:
            from mmd.dft.dft import get_xc
            self.Exc,self.Vxc = get_xc(self.grid,0.5*self.P,xcname=self.xcname)
            self.G = 2.*self.J + self.Vxc
        else:
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
        if self.xcname:
            # compute DFT energy --- need to combine at some point
            self.el_energy = (2*np.einsum('pq,qp',self.Core,self.P) +
                      2*np.einsum('pq,qp',self.J,self.P) + self.Exc)
        else:
            # regular ol' hartree fock. 
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

