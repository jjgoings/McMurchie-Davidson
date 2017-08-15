from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.linalg import expm

class RealTime(object):
    """Class for real-time routines"""
    def __init__(self,mol,numsteps=1000,stepsize=0.1,field=0.0001,pulse=None):
        self.mol = mol
        self.field = field
        self.stepsize = stepsize
        self.numSteps = numsteps
        self.time = np.arange(0,self.numSteps)*self.stepsize
        if pulse:
            self.pulse = pulse
        else:
            # zero pulse envelope
            self.pulse = lambda t: 0.0
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
        """Propagate in time using the second order explicit Magnus.
           See: Blanes, Sergio, and Fernando Casas. A concise introduction 
           to geometric numerical integration. Vol. 23. CRC Press, 2016.

           Magnus2 is Eq (4.61), page 128.
        """
        self.reset()
        self.mol.orthoDen()
        self.mol.orthoFock()
        h = -1j*self.stepsize
        for idx,time in enumerate((self.time)):
            if direction.lower() == 'x':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[0]))
            elif direction.lower() == 'y':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[1]))
            elif direction.lower() == 'z':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[2]))

            # record pulse envelope for later plotting, etc.
            self.shape.append(self.pulse(time))
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
            
            # density and Fock are done updating, wrap things up
            self.mol.unOrthoFock()    
            self.mol.unOrthoDen()    
            self.mol.computeEnergy()
            self.Energy.append(np.real(self.mol.energy))

    def Magnus4(self,direction='x'):
        """Propagate in time using the fourth order explicit Magnus.
           See: Blanes, Sergio, and Fernando Casas. A concise introduction 
           to geometric numerical integration. Vol. 23. CRC Press, 2016.

           Magnus4 is Eq (4.62), page 128.
        """
        self.reset()
        self.mol.orthoDen()
        self.mol.orthoFock()
        h = -1j*self.stepsize
        for idx,time in enumerate((self.time)):
            if direction.lower() == 'x':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[0]))
            elif direction.lower() == 'y':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[1]))
            elif direction.lower() == 'z':
                self.mol.computeDipole()
                self.dipole.append(np.real(self.mol.mu[2]))
            # record pulse envelope for later plotting, etc.
            self.shape.append(self.pulse(time))
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
            
            # density and Fock are done updating, wrap things up
            self.mol.unOrthoFock()    
            self.mol.unOrthoDen()    
            self.mol.computeEnergy()
            self.Energy.append(np.real(self.mol.energy))

    def addField(self,time,direction='x'):
        """ Add the electric dipole contribution to the Fock matrix,
            and then orthogonalize the results. The envelope (shape) of 
            the interaction with the electric field (self.pulse) needs 
            to be set externally in a job, since the desired pulse is 
            specific to each type of realtime simulation.

            self.pulse: function of time (t) that returns the envelope
                        amplitude at a given time. 
            Example: 
                def gaussian(t):
                    envelope = np.exp(-(t**2))
                    return envelope

                rt = RealTime(molecule, pulse=gaussian, field=0.001)
           
            The above example would set up a realtime simulations with
            the external field to have the gaussian envelope defined above 
            scaled by field=0.001.
        """

        shape = self.pulse(time) 

        if direction.lower() == 'x':
            self.mol.F += -self.field*shape*self.mol.M[0]
        elif direction.lower() == 'y':
            self.mol.F += -self.field*shape*self.mol.M[1]
        elif direction.lower() == 'z':
            self.mol.F += -self.field*shape*self.mol.M[2]
        self.mol.orthoFock()


