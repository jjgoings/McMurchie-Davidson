from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.linalg import expm
import sys
import itertools

class RealTime(object):
    """Class for real-time routines"""
    def __init__(self,mol,numsteps=1000,stepsize=0.1,field=0.0001,direction='x'):
        #self.SCF(doPrint=False,save=False)
        self.mol        = mol
        self.dipole     = []
        self.angmom     = []
        self.Energy     = []
        self.field = field
        self.stepsize = stepsize
        self.numSteps = numsteps
        self.time = np.arange(0,self.numSteps)*self.stepsize
        self.shape = []
        # remove from init?
        self.Magnus2(direction=direction)

    def Magnus2(self,direction='x'):
        self.mol.orthoDen()
        self.mol.orthoFock()
        h = -1j*self.stepsize
        for idx,time in enumerate(self.time):
            if direction.lower() == 'x':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.M[0]))
            elif direction.lower() == 'y':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.M[1]))
            elif direction.lower() == 'z':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.M[2]))
            self.addField(time,addstep=True,direction=direction)
            curDen  = np.copy(self.mol.PO)
      
            self.addField(time + 0.0*self.stepsize,direction=direction)
            k1 = h*self.mol.FO 
            U = expm(k1)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.updateFock()
            
            self.addField(time + 1.0*self.stepsize,direction=direction)
            L  = 0.5*(k1 + h*self.mol.FO)
            U  = expm(L)
            self.mol.PO = np.dot(U,np.dot(curDen,self.mol.adj(U))) 
            self.updateFock()
            
            self.mol.unOrthoFock()    
            self.mol.unOrthoDen()    
            self.mol.computeEnergy()
            self.Energy.append(np.real(self.mol.energy))
            print(self.mol.energy)

    def addField(self,time,addstep=False,direction='x'):
        if time == 0.0:
            shape = 1.0
        else:
            shape = 0.0
    
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

    def updateFock(self):
        self.mol.unOrthoDen()
        self.mol.buildFock()
        self.mol.orthoFock()


