from __future__ import division
import numpy as np
from scipy.linalg import eigh

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
            
    def RHF(self):
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
                    print "Error", np.linalg.norm(error)
                    print "E(SCF)    = ", "{0:.12f}".format(self.mol.energy.real)+ \
                          " in "+str(step)+" iterations"
                    print " RMS(P)  = ", "{0:.2e}".format(self.P_RMS.real)
                    print " dE(SCF) = ", "{0:.2e}".format(self.delta_energy.real)
                    self.computeDipole()
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
        self.mol.mu_x = -2*np.trace(np.dot(self.mol.P,self.mol.Mx)) + sum([x[0]*(x[1][0]-self.mol.gauge_origin[0]) for x in self.mol.atoms])  
        self.mol.mu_y = -2*np.trace(np.dot(self.mol.P,self.mol.My)) + sum([x[0]*(x[1][1]-self.mol.gauge_origin[1]) for x in self.mol.atoms])  
        self.mol.mu_z = -2*np.trace(np.dot(self.mol.P,self.mol.Mz)) + sum([x[0]*(x[1][2]-self.mol.gauge_origin[2]) for x in self.mol.atoms])  
        # to debye
        self.mol.mu_x *= 2.541765
        self.mol.mu_y *= 2.541765
        self.mol.mu_z *= 2.541765
    
    def adj(self,x):
        return np.conjugate(x).T       
    
    def comm(self,A,B):
        return np.dot(A,B) - np.dot(B,A)
    
