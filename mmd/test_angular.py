import numpy as np
from scipy.integrate import quad,nquad,tplquad
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from integrals import *
import matplotlib.pyplot as plt

def G(r,a,l1,A):
    rA   = np.power(r - A,l1)
    rAsq = np.power(r - A,2) 
    return rA*np.exp(-a*rAsq)

def gauss1d(r,a,l1,A,b,l2,B,n=0,london=False):
    G1   = G(r,a,l1,A) 
    G2   = G(r,b,l2,B) 
    if london:
        return G1*G2*np.power(r-B,n)
    else:
        return G1*G2*np.power(r,n)


def del1d(r,a,l1,A,b,l2,B):
    G1   = G(r,a,l1,A) 
    m = np.arange(r-0.1,r+0.1,0.00075)
    G2   = G(m,b,l2,B)
#    print spline(m,G2).derivatives(r)
    G2p = spline(m,G2).__call__(r,1)
    #print G(r,b,l2,B), G2p
    return G1*G2p

def numAngular(a,lmn1,A,b,lmn2,B,direction='x',london=False):
    Sx  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0],0))[0]
    Sy  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1],0))[0]
    Sz  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2],0))[0]
    Rx  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0],1,london))[0] 
    Ry  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1],1,london))[0] 
    Rz  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2],1,london))[0] 
    Dx  = quad(del1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0]))[0] 
    Dy  = quad(del1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1]))[0] 
    Dz  = quad(del1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2]))[0] 
    if direction.lower() == 'x':
        return -Sx*(Ry*Dz - Rz*Dy) 
    elif direction.lower() == 'y':
        return -Sy*(-Rx*Dz + Rz*Dx) 
    elif direction.lower() == 'z':
        return -Sz*(Rx*Dy - Ry*Dx) 


# works but way too slow...
#def numOverlap(a,lmn1,A,b,lmn2,B):
#   x1 = -10.0
#   x2 =  10.0 
#   return nquad(gauss3d,[(x1,x2),(x1,x2),(x1,x2)] , args=(a,lmn1,A,b,lmn2,B))
        
 

a = 0.05
lmn1 = (0,2,1)
A = np.asarray((0.2, 0.0, 18.0))

b = 0.02
lmn2 = (3,1,0)
B = np.asarray((0.0,-0.4,17.0))

n = (0,0,0)
C = np.asarray((0.0,0.0,0.0)) 

london=False
print angular(a,lmn1,A,b,lmn2,B,C,direction='x',london=london)
print numAngular(a,lmn1,A,b,lmn2,B,direction='x',london=london)

print angular(a,lmn1,A,b,lmn2,B,C,direction='y',london=london)
print numAngular(a,lmn1,A,b,lmn2,B,direction='y',london=london)


print angular(a,lmn1,A,b,lmn2,B,C,direction='z',london=london)
print numAngular(a,lmn1,A,b,lmn2,B,direction='z',london=london)

