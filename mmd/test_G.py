import numpy as np
import time
from skmonaco import mcquad, mcimport, mcmiser
from numpy.random import normal
import random
from scipy.integrate import fixed_quad,quadrature,nquad,tplquad,quad, simps,romberg
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from integrals import *
from hermite import *
from numeri import eri_integrand,eri_wintegrand,distribution 
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def G(r,a,l1,A):
    rA   = np.power(r - A,l1)
    rAsq = np.power(r - A,2)
    return rA*np.exp(-a*rAsq)

def gauss1d(r,a,l1,A,b,l2,B):
    G1   = G(r,a,l1,A)
    G2   = G(r,b,l2,B)
    return G1*G2

#def distribution(size,sp,Px,Py,Pz,sq,Qx,Qy,Qz):
#    x1 = normal(Px,sp,size=size)
#    y1 = normal(Py,sp,size=size)
#    z1 = normal(Pz,sp,size=size)
#    x2 = normal(Qx,sq,size=size)
#    y2 = normal(Qy,sq,size=size)
#    z2 = normal(Qz,sq,size=size)
#    return np.array((x1,y1,z1,x2,y2,z2)).T
#


   

#def eri_integrand(x,a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D,n1,n2):
#    val  = 1.0
#    val *= G(x[0],a,lmn1[0],A[0])
#    val *= G(x[1],a,lmn1[1],A[1])
#    val *= G(x[2],a,lmn1[2],A[2])
#
#    val *= G(x[0],b,lmn2[0],B[0])
#    val *= G(x[1],b,lmn2[1],B[1])
#    val *= G(x[2],b,lmn2[2],B[2])
#
#    val *= 1.0/np.sqrt(np.power(x[0] - x[3],2) + np.power(x[1]-x[4],2) + np.power(x[2]-x[5],2))
#
#    val *= G(x[3],c,lmn3[0],C[0])
#    val *= G(x[4],c,lmn3[1],C[1])
#    val *= G(x[5],c,lmn3[2],C[2])
#
#    val *= G(x[3],d,lmn4[0],D[0])
#    val *= G(x[4],d,lmn4[1],D[1])
#    val *= G(x[5],d,lmn4[2],D[2])
#
#    return val

a = 0.3
lmn1 = np.asarray((0,0,0)).astype(long)
A = np.asarray([0.0, 0.0, 0.0])

b = 0.35
lmn2 = np.asarray((1,0,0)).astype(long)
B = np.asarray([0.0,0.0,0.0])

c = 0.2
lmn3 = np.asarray((0,1,1)).astype(long)
C = np.asarray([0.0,0.0,1.4])

d = 0.22
lmn4 = np.asarray((0,0,0)).astype(long)
D = np.asarray([0.0,0.0,1.4])

#N1 = (0,1,2)
#N2 = (4,0,0)
N1 = (1,0,2)
N2 = (1,2,0)

gOrigin=np.array([43.0,-15.0,-80.0])

v1 = electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D,N1,N2,gOrigin)
print v1

N = 1e8 
#lower = -4.0
#upper = 4.0 

l1,m1,n1 = lmn1
l2,m2,n2 = lmn2
l3,m3,n3 = lmn3
l4,m4,n4 = lmn4

Ax,Ay,Az = A
Bx,By,Bz = B
Cx,Cy,Cz = C
Dx,Dy,Dz = D

P = (a*A + b*B)/(a + b)
Q = (c*C + d*D)/(c + d)

Px,Py,Pz = P
Qx,Qy,Qz = Q
p = a + b
q = c + d

nx1,ny1,nz1 = N1
nx2,ny2,nz2 = N2

gx = gOrigin[0]
gy = gOrigin[1]
gz = gOrigin[2]

sp = np.sqrt(1.0/(2.*(p)))
sq = np.sqrt(1.0/(2.*(q)))


#result, error = mcquad(lambda x: eri_integrand(x,a,l1,m1,n1,Ax,Ay,Az,b,l2,m2,n2,Bx,By,Bz,c,l3,m3,n3,Cx,Cy,Cz,d,l4,m4,n4,Dx,Dy,Dz,nx1,ny1,nz1,nx2,ny2,nz2),\
#                    xl=[lower]*6,xu=[upper]*6,npoints=N,nprocs=2,batch_size=1000000)

#result, error = mcmiser(lambda x: eri_integrand(x,a,l1,m1,n1,Ax,Ay,Az,b,l2,m2,n2,Bx,By,Bz,c,l3,m3,n3,Cx,Cy,Cz,d,l4,m4,n4,Dx,Dy,Dz,nx1,ny1,nz1,nx2,ny2,nz2),\
#                    xl=[lower]*6,xu=[upper]*6,npoints=N,nprocs=16)

t1 = time.time()
result, error = mcimport(eri_wintegrand,
                         npoints=N,
                         distribution=distribution,
                         args=(a,l1,m1,n1,Ax,Ay,Az,
                               b,l2,m2,n2,Bx,By,Bz,
                               c,l3,m3,n3,Cx,Cy,Cz,
                               d,l4,m4,n4,Dx,Dy,Dz,
                               nx1,ny1,nz1,nx2,ny2,nz2,
                               gx,gy,gz),
                         dist_kwargs={'sp':sp,'Px':Px,'Py':Py,'Pz':Pz,
                                      'sq':sq,'Qx':Qx,'Qy':Qy,'Qz':Qz},
                         nprocs=2,
                         batch_size=1e4)
t2 = time.time()

print result, error
print "Time: ", t2-t1, " seconds"


