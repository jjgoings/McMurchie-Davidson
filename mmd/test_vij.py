import numpy as np
from scipy.integrate import fixed_quad,quad,nquad,tplquad
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from integrals import *
import matplotlib.pyplot as plt

def G(r,a,l1,A):
    rA   = np.power(r - A,l1)
    rAsq = np.power(r - A,2) 
    return rA*np.exp(-a*rAsq)

def gauss1d(r,a,l1,A,b,l2,B):
    G1   = G(r,a,l1,A) 
    G2   = G(r,b,l2,B) 
    return G1*G2

def coulomb1d(r,t,a,l1,A,b,l2,B,C,n):
    G1   = G(r,a,l1,A) 
    G2   = G(r,b,l2,B) 
    Gc   = G(r,np.power(t,2),0,C)
    return G1*G2*Gc*np.power(r,n)

def V(t,a,lmn1,A,b,lmn2,B,C,n):
    val = 1.0/np.sqrt(np.pi)
    val *= quad(coulomb1d, -np.inf, np.inf, args=(t,a,lmn1[0],A[0],b,lmn2[0],B[0],C[0],n[0]))[0] 
    val *= quad(coulomb1d, -np.inf, np.inf, args=(t,a,lmn1[1],A[1],b,lmn2[1],B[1],C[1],n[1]))[0] 
    val *= quad(coulomb1d, -np.inf, np.inf, args=(t,a,lmn1[2],A[2],b,lmn2[2],B[2],C[2],n[2]))[0] 
    return val


def numNuclear(a,lmn1,A,b,lmn2,B,C,n=(0,0,0)):
    val = 1.0
    t   = np.inf 
    val *= quad(V, -t, t, args=(a,lmn1,A,b,lmn2,B,C,n))[0] 
    return val


a = 0.1
lmn1 = (2,0,1)
A = np.asarray([0.2, 0.0, 11.0])

b = 0.002
lmn2 = (0,3,2)
B = np.asarray([0.1,0.8,10.4])

origin = np.asarray([0.15, 0.5, 10.00])

v1 = nuclear_attraction(a,lmn1,A,b,lmn2,B,origin,n=(1,3,2))
v2 = numNuclear(a,lmn1,A,b,lmn2,B,origin,n=(1,3,2))

print v1
print v2

#t = np.arange(-2,2,0.01)
#v = []
#for i in t:
#   v.append(V(i,a,lmn1,A,b,lmn2,B,origin))

#plt.plot(t,np.asarray(v))
#plt.show()



