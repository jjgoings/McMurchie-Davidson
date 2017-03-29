import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.linalg import expm

cpdef np.ndarray[complex,ndim=2] buildFock(np.ndarray[double,ndim=4] TwoE,np.ndarray[complex,ndim=2] P,np.ndarray[double,ndim=2] Core):
    J = np.einsum('pqrs,rs->pq', TwoE,P)
    K = np.einsum('prqs,rs->pq', TwoE,P)
    G = 2.*J - K
    return Core + G

cpdef np.ndarray[complex,ndim=2] orthoFock(np.ndarray[double,ndim=2] X, np.ndarray[complex,ndim=2] F):
    # returns FO
    return np.dot(X.T,np.dot(F,X))

def unOrthoFock(U,FO):
    # returns F
    return np.dot(U.T,np.dot(FO,U))

def orthoDen(U,P):
    # returns PO
    return np.dot(U,np.dot(P,U.T))

def unOrthoDen(X,PO):
    # returns P
    return np.dot(X,np.dot(PO,X.T))

def adj(x):
    return np.conjugate(x).T       

def comm(A,B):
    return np.dot(A,B) - np.dot(B,A)

def updateFock(X,PO,TwoE,Core):
    # returns F
    #return orthoFock(X,buildFock(TwoE,unOrthoDen(X,PO),Core))
    return buildFock(TwoE,unOrthoDen(X,PO),Core)

def addField(time,X,F,Mx,My,Mz,field,direction='x'):
    # returns FO with field included
    if time == 0.0:
        shape = 1.0
    else:
        shape = 0.0

    if direction.lower() == 'x':
        F += -field*shape*Mx
    elif direction.lower() == 'y':
        F += -field*shape*My
    elif direction.lower() == 'z':
        F += -field*shape*Mz

    return orthoFock(X,F) 


cpdef updateM4(double time,str direction, double stepsize, np.ndarray[complex,ndim=2] PO,np.ndarray[complex,ndim=2] F,np.ndarray[double,ndim=2] X,np.ndarray[double,ndim=4] TwoE,np.ndarray[double,ndim=2] Core, np.ndarray[double,ndim=2] Mx,np.ndarray[double,ndim=2] My,np.ndarray[double,ndim=2] Mz, double field):
    cdef complex h = -1j*stepsize
    cpdef np.ndarray[complex,ndim=2] curDen  = np.copy(PO)

    FO = addField(time + 0.0*stepsize,X,F,Mx,My,Mz,field,direction=direction)
    cpdef np.ndarray[complex,ndim=2] k1 = h*FO 
    cpdef np.ndarray[complex,ndim=2] Q1 = k1
    cpdef np.ndarray[complex,ndim=2] U = expm(0.5*Q1)
    PO = np.dot(U,np.dot(curDen,adj(U))) 
    F = updateFock(X,PO,TwoE,Core)
    
    FO = addField(time + 0.5*stepsize,X,F,Mx,My,Mz,field,direction=direction)
    cpdef np.ndarray[complex,ndim=2] k2 = h*FO
    cpdef np.ndarray[complex,ndim=2] Q2 = k2 - k1
    U = expm(0.5*Q1 + 0.25*Q2)
    PO = np.dot(U,np.dot(curDen,adj(U))) 
    F = updateFock(X,PO,TwoE,Core)

    FO = addField(time + 0.5*stepsize,X,F,Mx,My,Mz,field,direction=direction)
    cpdef np.ndarray[complex,ndim=2] k3 = h*FO
    cpdef np.ndarray[complex,ndim=2] Q3 = k3 - k2
    U = expm(Q1 + Q2)
    PO = np.dot(U,np.dot(curDen,adj(U))) 
    F = updateFock(X,PO,TwoE,Core)

    FO = addField(time + 1.0*stepsize,X,F,Mx,My,Mz,field,direction=direction)
    cpdef np.ndarray[complex,ndim=2] k4 = h*FO
    cpdef np.ndarray[complex,ndim=2] Q4 = k4 - 2*k2 + k1
    cpdef np.ndarray[complex,ndim=2 ] L  = 0.5*Q1 + 0.25*Q2 + (1/3.)*Q3 - (1/24.)*Q4 -(1/48.)*comm(Q1,Q2)
    U  = expm(L)
    PO = np.dot(U,np.dot(curDen,adj(U))) 
    F = updateFock(X,PO,TwoE,Core)
   
    FO = addField(time + 0.5*stepsize,X,F,Mx,My,Mz,field,direction=direction)
    cpdef np.ndarray[complex,ndim=2] k5 = h*FO
    cpdef np.ndarray[complex,ndim=2] Q5 = k5 - k2 
    L  = Q1 + Q2 + (2/3.)*Q3 + (1/6.)*Q4 - (1/6.)*comm(Q1,Q2)
    U  = expm(L)
    PO = np.dot(U,np.dot(curDen,adj(U))) 
    F = updateFock(X,PO,TwoE,Core)

    FO = addField(time + 1.0*stepsize,X,F,Mx,My,Mz,field,direction=direction)
    cpdef np.ndarray[complex,ndim=2] k6 = h*FO
    cpdef np.ndarray[complex,ndim=2] Q6 = k6 -2*k2 + k1
    L  = Q1 + Q2 + (2/3.)*Q5 + (1/6.)*Q6 -(1/6.)*comm(Q1, (Q2 - Q3 + Q5 + 0.5*Q6))

    U  = expm(L)
    PO = np.dot(U,np.dot(curDen,adj(U))) 
    F = updateFock(X,PO,TwoE,Core)

    FO = orthoFock(X,F)
    return FO, PO

    
         




