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

def gauss1d(r,a,l1,A,b,l2,B,n=0):
    G1   = G(r,a,l1,A) 
    G2   = G(r,b,l2,B) 
    return G1*G2*np.power(r,n)


def kinetic1d(r,a,l1,A,b,l2,B,n):
    G1   = G(r,a,l1,A) 
    m = np.arange(r-0.1,r+0.1,0.00075)
    G2   = G(m,b,l2,B)
#    print spline(m,G2).derivatives(r)
    G2p = spline(m,G2).__call__(r,2)
    #print G(r,b,l2,B), G2p
    return G1*G2p*np.power(r,n)

def gauss3d(z,y,x,a,lmn1,A,b,lmn2,B):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    xA   = np.power(x - A[0],l1)
    yA   = np.power(y - A[1],m1)
    zA   = np.power(z - A[2],n1)
    xAsq = np.power(x - A[0],2) 
    yAsq = np.power(y - A[1],2) 
    zAsq = np.power(z - A[2],2) 
    xB   = np.power(x - B[0],l2)
    yB   = np.power(y - B[1],m2)
    zB   = np.power(z - B[2],n2)
    xBsq = np.power(x - B[0],2) 
    yBsq = np.power(y - B[1],2) 
    zBsq = np.power(z - B[2],2) 
    G1 = xA*yA*zA*np.exp(-a*(xAsq + yAsq + zAsq))
    G2 = xB*yB*zB*np.exp(-b*(xBsq + yBsq + zBsq))
    return G1*G2 

def numNuclear(a,lmn1,A,b,lmn2,B):
    val = 1.0
    val *= quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0]))[0] 
    val *= quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1]))[0] 
    val *= quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2]))[0] 
    return val

def numOverlap(a,lmn1,A,b,lmn2,B,n):
    val = 1.0
    val *= quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0],n[0]))[0] 
    val *= quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1],n[1]))[0] 
    val *= quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2],n[2]))[0] 
    return val

def numKinetic(a,lmn1,A,b,lmn2,B,n):
    val = 0.0 
    Sx  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0],n[0]))[0] 
    Sy  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1],n[1]))[0] 
    Sz  = quad(gauss1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2],n[2]))[0] 
    Tx  = quad(kinetic1d, -np.inf, np.inf, args=(a,lmn1[0],A[0],b,lmn2[0],B[0],n[0]))[0] 
    Ty  = quad(kinetic1d, -np.inf, np.inf, args=(a,lmn1[1],A[1],b,lmn2[1],B[1],n[1]))[0] 
    Tz  = quad(kinetic1d, -np.inf, np.inf, args=(a,lmn1[2],A[2],b,lmn2[2],B[2],n[2]))[0] 

    return -0.5*(Tx*Sy*Sz + Ty*Sz*Sx + Tz*Sx*Sy)

# works but way too slow...
#def numOverlap(a,lmn1,A,b,lmn2,B):
#   x1 = -10.0
#   x2 =  10.0 
#   return nquad(gauss3d,[(x1,x2),(x1,x2),(x1,x2)] , args=(a,lmn1,A,b,lmn2,B))
        
 

a = 0.05
lmn1 = (0,2,1)
A = (0.2, 0.0, 1.0)

b = 0.02
lmn2 = (3,1,0)
B = (0.0,-0.4,0.0)

n = (4,2,4)
C =np.array([0,0,0])

print overlap(a,lmn1,A,b,lmn2,B,n)
print numOverlap(a,lmn1,A,b,lmn2,B,n)
print kinetic(a,lmn1,A,b,lmn2,B,C,n)
print numKinetic(a,lmn1,A,b,lmn2,B,n)




