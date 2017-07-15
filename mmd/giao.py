from __future__ import division
import numpy as np
from mmd.integrals.onee import S,T,V,RxDel
from mmd.integrals.twoe import do2eGIAO

class GIAO(object):
    """Class with additional routines for creation of (some) london orbitals
       relevant to the gauge-invariant mangetic dipole moment opperator. 
    """
    def build_GIAO(self):
        """Routine to build some GIAO integrals"""
        self.GIAO_one_electron_integrals()
        self.GIAO_two_electron_integrals()

    def GIAO_one_electron_integrals(self):
        """Routine to compute some GIAO one electron integrals"""
        N = self.nbasis

        #GIAO overlap
        self.Sb = np.zeros((3,N,N))

        # derivative of one-electron GIAO integrals wrt B at B = 0.
        self.rH = np.zeros((3,N,N)) 

        # London Angular momentum L_N
        self.Ln = np.zeros((3,N,N))

        # holds total dH/dB = 0.5*Ln + rH
        self.dhdb = np.zeros((3,N,N))

        #print "GIAO one-electron integrals"
        for i in (range(N)):
            for j in range(N):
                #QAB matrix elements
                XAB = self.bfs[i].origin[0] - self.bfs[j].origin[0]
                YAB = self.bfs[i].origin[1] - self.bfs[j].origin[1]
                ZAB = self.bfs[i].origin[2] - self.bfs[j].origin[2]
                # GIAO T
                self.rH[0,i,j] = T(self.bfs[i],self.bfs[j],n=(1,0,0),gOrigin=self.gauge_origin)
                self.rH[1,i,j] = T(self.bfs[i],self.bfs[j],n=(0,1,0),gOrigin=self.gauge_origin)
                self.rH[2,i,j] = T(self.bfs[i],self.bfs[j],n=(0,0,1),gOrigin=self.gauge_origin)

                for atom in self.atoms:
                    # GIAO V
                    self.rH[0,i,j] += -atom.charge*V(self.bfs[i],self.bfs[j],atom.origin,n=(1,0,0),gOrigin=self.gauge_origin)
                    self.rH[1,i,j] += -atom.charge*V(self.bfs[i],self.bfs[j],atom.origin,n=(0,1,0),gOrigin=self.gauge_origin)
                    self.rH[2,i,j] += -atom.charge*V(self.bfs[i],self.bfs[j],atom.origin,n=(0,0,1),gOrigin=self.gauge_origin)

                # Some temp copies for mult with QAB matrix 
                xH = self.rH[0,i,j]
                yH = self.rH[1,i,j]
                zH = self.rH[2,i,j]
               
                # add QAB contribution 
                self.rH[0,i,j] = 0.5*(-ZAB*yH + YAB*zH)
                self.rH[1,i,j] = 0.5*( ZAB*xH - XAB*zH)
                self.rH[2,i,j] = 0.5*(-YAB*xH + XAB*yH)

                # add QAB contribution for overlaps 
                #C = np.asarray([0,0,0])
                Rx = S(self.bfs[i],self.bfs[j],n=(1,0,0),gOrigin=self.gauge_origin)
                Ry = S(self.bfs[i],self.bfs[j],n=(0,1,0),gOrigin=self.gauge_origin)
                Rz = S(self.bfs[i],self.bfs[j],n=(0,0,1),gOrigin=self.gauge_origin)
                self.Sb[0,i,j] = 0.5*(-ZAB*Ry + YAB*Rz)
                self.Sb[1,i,j] = 0.5*( ZAB*Rx - XAB*Rz)
                self.Sb[2,i,j] = 0.5*(-YAB*Rx + XAB*Ry)

                # now do Angular London Momentum
                self.Ln[0,i,j] = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'x',london=True)
                self.Ln[1,i,j] = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'y',london=True)
                self.Ln[2,i,j] = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'z',london=True)

        # below gives dH/dB accoriding to dalton
        self.dhdb[:] = 0.5*self.Ln[:] + self.rH[:]

    def GIAO_two_electron_integrals(self):
        """Routine to setup and compute some GIAO two-electron integrals"""
        N = self.nbasis
        self.GR1 = np.zeros((3,N,N,N,N))  
        self.GR2 = np.zeros((3,N,N,N,N))  
        self.dgdb = np.zeros((3,N,N,N,N))  
        #print "GIAO two-electron integrals"
        self.dgdb = do2eGIAO(N,self.GR1,self.GR2,self.dgdb,self.bfs,self.gauge_origin)
        self.dgdb = np.asarray(self.dgdb)

    def buildL(self,direction='x'):
         """Build the GIAO magnetic dipole moment operator.
            Returns complex float in self.LN, which depends on the input 
            axis (x,y,z). 
            self.LN = L_GIAO = 2*<dH/dB> + <dG/dB> - 2*<dS/dB> evaluated at B=0.
            This expression is the similar to nuclear gradient terms, etc.
         """
         # W1 is equivalent to P*F*P
         #self.orthoFock()
         #E,CO = np.linalg.eigh(self.FO)
         #C      = np.dot(self.X,CO)
         #W1 = np.zeros((self.nbasis,self.nbasis),dtype='complex')
         #for mu in range(self.nbasis):
         #    for nu in range(self.nbasis):
         #        for i in range(self.nocc):
         #            W1[mu,nu] += E[i]*np.conjugate(C[mu,i])*C[nu,i]
         #print E

         if direction.lower() == 'x':
             dHdB = 1j*self.dhdb[0]
             dGdB = 1j*self.dgdb[0]
             dSdB = 1j*self.Sb[0]
         elif direction.lower() == 'y':
             dHdB = 1j*self.dhdb[1]
             dGdB = 1j*self.dgdb[1]
             dSdB = 1j*self.Sb[1]
         elif direction.lower() == 'z':
             dHdB = 1j*self.dhdb[2]
             dGdB = 1j*self.dgdb[2]
             dSdB = 1j*self.Sb[2]

         J = np.einsum('pqrs,sr->pq', dGdB,self.P)
         K = np.einsum('psqr,sr->pq', dGdB,self.P)
         G = 2.*J - K
         F = dHdB + G
         self.LN = np.einsum('pq,qp',self.P,F + dHdB) # Correct for GS
         PFP = np.dot(self.P,np.dot(self.F,self.P))
         W = PFP
         self.LN -= 2*np.einsum('pq,qp',dSdB,W)



