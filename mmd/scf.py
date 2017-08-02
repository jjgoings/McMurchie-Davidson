from __future__ import division
from __future__ import print_function
import numpy as np
from numpy.linalg import multi_dot as dot
from mmd.integrals.twoe import ERI
from mmd.integrals.fock import formPT

class SCF(object):
    """SCF methods and routines for molecule object"""
    def RHF(self,doPrint=True,DIIS=True,direct=False):
        """Routine to compute the RHF energy for a closed shell molecule"""
        self.is_converged = False
        self.delta_energy = 1e20
        self.P_RMS        = 1e20
        self.P_old        = np.zeros((self.nbasis,self.nbasis),dtype='complex')
        self.maxiter = 50
        self.direct = direct
        if self.direct:
            self.incFockRst = False
        self.build(self.direct) # build integrals

        self.P = self.P_old
        self.F = self.Core.astype('complex')

        if DIIS:
            self.fockSet = []
            self.errorSet = []

        for step in range(self.maxiter):
            if step > 0:
                self.F_old      = self.F
                energy_old = self.energy
                # need old P for incremental Fock build
                self.buildFock()
                # now update old P
                self.P_old      = self.P

                # note that extrapolated Fock cannot be used with incrm. Fock
                # build. We make a copy to get the new density only.
                if DIIS: 
                    F_diis = self.updateDIIS(self.F,self.P)
                    self.FO = np.dot(self.X.T,np.dot(F_diis,self.X)) 
            if not DIIS or step == 0:
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
            SPF = self.adj(FPS)
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
        if self.direct:
           # N = self.nbasis
           # # perturbation tensor and density difference
           # self.G = np.zeros((N,N),dtype='complex')
           # dP = self.P - self.P_old
           # # Comments from LibInt hartreefock++
           # #  1) each shell set of integrals contributes up to 6 shell sets of
           # #  the Fock matrix:
           # #     F(a,b) += (ab|cd) * D(c,d)
           # #     F(c,d) += (ab|cd) * D(a,b)
           # #     F(b,d) -= 1/4 * (ab|cd) * D(a,c)
           # #     F(b,c) -= 1/4 * (ab|cd) * D(a,d)
           # #     F(a,c) -= 1/4 * (ab|cd) * D(b,d)
           # #     F(a,d) -= 1/4 * (ab|cd) * D(b,c)
           # #  2) each permutationally-unique integral (shell set) must be
           # #  scaled by its degeneracy,
           # #     i.e. the number of the integrals/sets equivalent to it
           # #  3) the end result must be symmetrized
           # for i in range(N):
           #     for j in range(i+1):
           #         ij = (i*(i+1)//2 + j)
           #         for k in range(N):
           #             for l in range(k+1):
           #                 kl = (k*(k+1)//2 + l)
           #                 if ij >= kl:
           #                     # use cauchy-schwarz to screen
           #                     bound = (np.sqrt(self.screen[ij])
           #                             *np.sqrt(self.screen[kl]))
           #                     # screen based on contr. with density diff
           #                     dmax = np.max(np.abs([4*dP[i,j],
           #                                    4*dP[k,l],
           #                                      dP[i,k],
           #                                      dP[i,l],
           #                                      dP[j,k],
           #                                      dP[j,l]]))
           #                     bound *= dmax
           #                     if bound < 1e-10:
           #                         continue
           #                     else:
           #                         # work out degeneracy scaling
           #                         s12_deg = 1.0 if (i == j) else 2.0
           #                         s34_deg = 1.0 if (k == l) else 2.0
           #                         if i == k:
           #                             if j == l:
           #                                 s12_34_deg = 1.0
           #                             else:
           #                                 s12_34_deg = 2.0
           #                         else:
           #                             s12_34_deg = 2.0

           #                         s1234_deg = s12_deg * s34_deg * s12_34_deg

           #                         #FIXME: should use integrals from screen
           #                         # rather than re-compute the values
           #                         eri = s1234_deg*ERI(self.bfs[i],self.bfs[j],
           #                                             self.bfs[k],self.bfs[l])

           #                        # # See Almlof, Faegri, Korsell, 1981
           #                        # # Coulomb, Eq (4a,4b) of Korsell, 1981 
           #                        # self.F[i,j] += self.P[k,l]*eri
           #                        # self.F[k,l] += self.P[i,j]*eri
           #                        # # Exchange, Eq (5) of Korsell, 1981 
           #                        # self.F[i,k] += -0.25*self.P[j,l]*eri
           #                        # self.F[j,l] += -0.25*self.P[i,k]*eri
           #                        # self.F[i,l] += -0.25*self.P[j,k]*eri
           #                        # self.F[k,j] += -0.25*self.P[i,l]*eri
           #                         # See Almlof, Faegri, Korsell, 1981
           #                         # Coulomb, Eq (4a,4b) of Korsell, 1981 
           #                         self.G[i,j] += dP[k,l]*eri
           #                         self.G[k,l] += dP[i,j]*eri
           #                         # Exchange, Eq (5) of Korsell, 1981 
           #                         self.G[i,k] += -0.25*dP[j,l]*eri
           #                         self.G[j,l] += -0.25*dP[i,k]*eri
           #                         self.G[i,l] += -0.25*dP[j,k]*eri
           #                         self.G[k,j] += -0.25*dP[i,l]*eri
           #                     
            if self.incFockRst:
                self.G = formPT(self.P,np.zeros_like(self.P),self.bfs,self.nbasis,self.screen,1e-12)
                self.G = 0.5*(self.G + self.G.T) 
                self.F = self.Core.astype('complex') + self.G
            else:
                self.G = formPT(self.P,self.P_old,self.bfs,self.nbasis,self.screen,1e-10)
                self.G = 0.5*(self.G + self.G.T) 
                self.F = self.F_old + self.G

        else:
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

    def updateDIIS(self,F,P):
        FPS =   dot([F,P,self.S])
        SPF =   self.adj(FPS) 
        # error must be in orthonormal basis
        error = dot([self.X,FPS-SPF,self.X]) 
        self.fockSet.append(self.F)
        self.errorSet.append(error) 
        numFock = len(self.fockSet)
        # limit subspace, hardcoded for now
        if numFock > 8:
            del self.fockSet[0] 
            del self.errorSet[0] 
            numFock -= 1
        B = np.zeros((numFock + 1,numFock + 1)) 
        B[-1,:] = B[:,-1] = -1.0
        B[-1,-1] = 0.0
        # B is symmetric
        for i in range(numFock):
            for j in range(i+1):
                B[i,j] = B[j,i] = \
                    np.real(np.trace(np.dot(self.adj(self.errorSet[i]),
                                                     self.errorSet[j])))
        residual = np.zeros(numFock + 1)
        residual[-1] = -1.0
        weights = np.linalg.solve(B,residual)

        # weights is 1 x numFock + 1, but first numFock values
        # should sum to one if we are doing DIIS correctly
        assert np.isclose(sum(weights[:-1]),1.0)

        F = np.zeros((self.nbasis,self.nbasis),dtype='complex')
        for i, Fock in enumerate(self.fockSet):
            F += weights[i] * Fock

        return F 

