from __future__ import division
import data 
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport gammainc 


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERI(object a,object b,object c,object d):
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
                                                d.exps[jd],d.shell,d.origin)
    return eri

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double E(int i,int j,int t,double Qx,double a,double b):
    p = a + b
    q = a*b/p
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
cpdef double electron_repulsion(double a, tuple lmn1,np.ndarray A,double b,tuple lmn2, np.ndarray B,double c,tuple lmn3,np.ndarray C,double d,tuple lmn4,np.ndarray D):
    cdef int l1, m1, n1
    cdef int l2, m2, n2
    cdef int l3, m3, n3
    cdef int l4, m4, n4
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    cdef double p = a+b
    cdef double q = c+d
    cdef double alpha = p*q/(p+q)
    cdef np.ndarray P = gaussian_product_center(a,A,b,B)
    cdef np.ndarray Q = gaussian_product_center(c,C,d,D)
    cdef double RPQ = np.linalg.norm(P-Q)

    cdef int t,u,v,tau,nu,phi
    cdef double val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val += E(l1,l2,t,A[0]-B[0],a,b) * \
                                   E(m1,m2,u,A[1]-B[1],a,b) * \
                                   E(n1,n2,v,A[2]-B[2],a,b) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d) * \
                                   pow(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ) 

    val *= 2*pow(np.pi,2.5)/(p*q*sqrt(p+q)) 
    return val 

cpdef double boys(int m,double T):
    # pretty sure this works, tested a few cases vs wolfram alpha
    if abs(T) < 1e-12:
        return 1/(2*m + 1)
    else:
        return gammainc(m+0.5,T)*tgamma(m+0.5)/(2*pow(T,m+0.5))

cpdef np.ndarray gaussian_product_center(double a, np.ndarray A,double b, np.ndarray B):
    return (a*A+b*B)/(a+b)


