from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.linalg import expm
import sys
import itertools
import copy

class RealTime(object):
    """Class for real-time routines"""
    def __init__(self,mol,numsteps=1000,stepsize=0.1,field=0.0001):
        self.mol = mol
        self.field = field
        self.stepsize = stepsize
        self.numSteps = numsteps
        self.time = np.arange(0,self.numSteps)*self.stepsize
        self.reset()

    def reset(self):
        """Reset all time-dependent property arrays to empty, will also 
           re-do the SCF in order to set the reference back to ground state.
           This will likely need to be changed in the future.
        """
        self.mol.RHF(doPrint=False)
        self.dipole     = []
        self.angmom     = []
        self.Energy     = []
        self.shape = []

    def Magnus2(self,direction='x'):
        """Propagate in time using the second order explicit Magnus"""
        self.reset()
        self.mol.orthoDen()
        self.mol.orthoFock()
        h = -1j*self.stepsize
        for idx,time in enumerate(self.time):
            if direction.lower() == 'x':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[0]))
            elif direction.lower() == 'y':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[1]))
            elif direction.lower() == 'z':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[2]))
            self.addField(time,addstep=True,direction=direction)
            curDen  = np.copy(self.mol.PO)
      
            self.addField(time + 0.0*self.stepsize,direction=direction)
            k1 = h*self.mol.FO 
            U = expm(k1)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()
            
            self.addField(time + 1.0*self.stepsize,direction=direction)
            L  = 0.5*(k1 + h*self.mol.FO)
            U  = expm(L)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()
            
            self.mol.unOrthoFock()    
            self.mol.unOrthoDen()    
            self.mol.computeEnergy()
            self.Energy.append(np.real(self.mol.energy))

    def Magnus4(self,direction='x'):
        """Propagate in time using the fourth order explicit Magnus"""
        self.reset()
        self.mol.orthoDen()
        self.mol.orthoFock()
        h = -1j*self.stepsize
        for idx,time in enumerate(self.time):
            if direction.lower() == 'x':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[0]))
            elif direction.lower() == 'y':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[1]))
            elif direction.lower() == 'z':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[2]))
            self.addField(time,addstep=True,direction=direction)
            curDen  = np.copy(self.mol.PO)
     
            self.addField(time + 0.0*self.stepsize,direction=direction)
            k1 = h*self.mol.FO 
            Q1 = k1
            U = expm(0.5*Q1)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()
            
            self.addField(time + 0.5*self.stepsize,direction=direction)
            k2 = h*self.mol.FO
            Q2 = k2 - k1
            U = expm(0.5*Q1 + 0.25*Q2)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()

            self.addField(time + 0.5*self.stepsize,direction=direction)
            k3 = h*self.mol.FO
            Q3 = k3 - k2
            U = expm(Q1 + Q2)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()

            self.addField(time + 1.0*self.stepsize,direction=direction)
            k4 = h*self.mol.FO
            Q4 = k4 - 2*k2 + k1
            L  = 0.5*Q1 + 0.25*Q2 + (1/3.)*Q3 - (1/24.)*Q4
            L += -(1/48.)*self.mol.comm(Q1,Q2)
            U  = expm(L)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()
           
            self.addField(time + 0.5*self.stepsize,direction=direction)
            k5 = h*self.mol.FO
            Q5 = k5 - k2 
            L  = Q1 + Q2 + (2/3.)*Q3 + (1/6.)*Q4 - (1/6.)*self.mol.comm(Q1,Q2)
            U  = expm(L)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()
 
            self.addField(time + 1.0*self.stepsize,direction=direction)
            k6 = h*self.mol.FO
            Q6 = k6 -2*k2 + k1
            L  = Q1 + Q2 + (2/3.)*Q5 + (1/6.)*Q6
            L += -(1/6.)*self.mol.comm(Q1, (Q2 - Q3 + Q5 + 0.5*Q6))

            U  = expm(L)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.mol.updateFock()
            
            self.mol.unOrthoFock()    
            self.mol.unOrthoDen()    
            self.mol.computeEnergy()
            self.Energy.append(np.real(self.mol.energy))

    def addField(self,time,addstep=False,direction='x'):
#        if time == 0.0:
#            shape = 1.0
#        else:
#            shape = 0.0
        t2 = 0.0 
        sigma2 = self.stepsize
        shape = (1.0/(sigma2*np.sqrt(2*np.pi)))*np.exp(-((time-t2)**2)/sigma2)

        if addstep:
            self.shape.append(shape)
        else:
            if direction.lower() == 'x':
                self.mol.F += -self.field*shape*self.mol.M[0]
            elif direction.lower() == 'y':
                self.mol.F += -self.field*shape*self.mol.M[1]
            elif direction.lower() == 'z':
                self.mol.F += -self.field*shape*self.mol.M[2]
            self.mol.orthoFock()


