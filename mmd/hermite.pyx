from __future__ import division
import data 
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport gammainc, hyp1f1 

cdef double pi = 3.141592653589793238462643383279


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERI(object a,object b,object c,object d, tuple n1 = (0,0,0), tuple n2 = (0,0,0), gOrigin = np.zeros((3))):
    cdef double eri = 0.0
    cdef int ja, jb, jc, jd
    cdef double ca, cb, cc, cd
    for ja, ca in enumerate(a.coefs):
        for jb, cb in enumerate(b.coefs):
            for jc, cc in enumerate(c.coefs):
                for jd, cd in enumerate(d.coefs):
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                             ca*cb*cc*cd*\
                             electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                                b.exps[jb],b.shell,b.origin,\
                                                c.exps[jc],c.shell,c.origin,\
                                                d.exps[jd],d.shell,d.origin,\
                                                n1,n2, gOrigin)
    return eri

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double E(int i,int j,int t,double Qx,double a,double b, int n = 0, double Ax = 0.0):
    p = a + b
    q = a*b/p
    if n == 0:
        if (t < 0) or (t > (i + j)):
            return 0.0
        elif i == j == t == 0:
            return exp(-q*Qx*Qx)
        elif j == 0:
            return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - (q*Qx/a)*E(i-1,j,t,Qx,a,b) + \
                   (t+1)*E(i-1,j,t+1,Qx,a,b)
        else:
            return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + (q*Qx/b)*E(i,j-1,t,Qx,a,b) + \
                   (t+1)*E(i,j-1,t+1,Qx,a,b)
    else:
        return E(i+1,j,t,Qx,a,b,n-1,Ax) + Ax*E(i,j,t,Qx,a,b,n-1,Ax)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double R(int t,int u,int v,int n, double p,double PCx, double PCy, double PCz, double RPC):
    cdef double T = p*RPC*RPC
    cdef double val = 0.0
    if t == u == v == 0:
        val += pow(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double electron_repulsion(double a, np.ndarray[long,mode='c',ndim=1] lmn1,np.ndarray[double,mode='c',ndim=1] A,double b,np.ndarray[long,mode='c',ndim=1] lmn2, np.ndarray[double,mode='c',ndim=1] B,double c,np.ndarray[long,mode='c',ndim=1] lmn3,np.ndarray[double,mode='c',ndim=1] C,double d,np.ndarray[long,mode='c',ndim=1] lmn4,np.ndarray[double,mode='c',ndim=1] D, tuple r1, tuple r2, np.ndarray[double,mode='c',ndim=1] gOrigin):
    cdef int l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
    cdef int l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef int l3 = lmn3[0], m3 = lmn3[1], n3 = lmn3[2]
    cdef int l4 = lmn4[0], m4 = lmn4[1], n4 = lmn4[2]
    cdef double p = a+b
    cdef double q = c+d
    cdef double alpha = p*q/(p+q)
    cdef double Px = (a*A[0] + b*B[0])/p
    cdef double Py = (a*A[1] + b*B[1])/p
    cdef double Pz = (a*A[2] + b*B[2])/p
    cdef double Qx = (c*C[0] + d*D[0])/q
    cdef double Qy = (c*C[1] + d*D[1])/q
    cdef double Qz = (c*C[2] + d*D[2])/q
    #cdef double RPQ = np.linalg.norm(P-Q)
    cdef double RPQ = sqrt(pow(Px-Qx,2) + \
                           pow(Py-Qy,2) + \
                           pow(Pz-Qz,2)) 

    cdef int t,u,v,tau,nu,phi
    cdef double val = 0.0
    cdef int r1x = r1[0], r1y = r1[1], r1z = r1[2]
    cdef int r2x = r2[0], r2y = r2[1], r2z = r2[2]
    for t in range(l1+l2+1+r1x):
        for u in range(m1+m2+1+r1y):
            for v in range(n1+n2+1+r1z):
                for tau in range(l3+l4+1+r2x):
                    for nu in range(m3+m4+1+r2y):
                        for phi in range(n3+n4+1+r2z):
                            val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                   E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                   E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                   pow(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 

    val *= 2*pow(pi,2.5)/(p*q*sqrt(p+q)) 
    return val 

@cython.cdivision(True)
cdef double boys(double m,double T):
    # pretty sure this works, tested a few cases vs wolfram alpha
    #if abs(T) < 1e-12:
    #    return 1/(2*m + 1)
    #else:
    #    return gammainc(m+0.5,T)*tgamma(m+0.5)/(2*pow(T,m+0.5))
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

#@cython.cdivision(True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef np.ndarray[double,mode='c',ndim=1] gaussian_product_center(double a, np.ndarray[double,mode='c',ndim=1] A,double b, np.ndarray[double,mode='c',ndim=1] B):
#    return (a*A+b*B)/(a+b)


