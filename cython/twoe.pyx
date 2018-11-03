from __future__ import division
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
include "util.pxi"
include "basis.pxi"

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double [:,:,:,:] doERIs(long N,double [:,:,:,:] TwoE, list bfs):
    cdef:
        long i,j,k,l,ij,kl
    for i in (range(N)):
        for j in range(i+1):
            ij = (i*(i+1)//2 + j)
            for k in range(N):
                for l in range(k+1):
                    kl = (k*(k+1)//2 + l)
                    if ij >= kl:
                       val = ERI(bfs[i],bfs[j],bfs[k],bfs[l])
                       TwoE[i,j,k,l] = val
                       TwoE[k,l,i,j] = val
                       TwoE[j,i,l,k] = val
                       TwoE[l,k,j,i] = val
                       TwoE[j,i,k,l] = val
                       TwoE[l,k,i,j] = val
                       TwoE[i,j,l,k] = val
                       TwoE[k,l,j,i] = val
    return TwoE


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERI(Basis a, Basis b, Basis c, Basis d):
    cdef double eri = 0.0
    cdef long ja, jb, jc, jd
    cdef double ca, cb, cc, cd
    for ja in range(a.num_exps):
        for jb in range(b.num_exps):
            for jc in range(c.num_exps):
                for jd in range(d.num_exps):
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                             a.coefs[ja]*b.coefs[jb]*c.coefs[jc]*d.coefs[jd]*\
                             electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                                b.exps[jb],b.shell,b.origin,\
                                                c.exps[jc],c.shell,c.origin,\
                                                d.exps[jd],d.shell,d.origin)
    return eri

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double electron_repulsion(double a, long *lmn1, double *A, double b, long *lmn2, double *B,double c, long *lmn3, double *C,double d, long *lmn4, double *D):
    cdef:
        long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
        long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
        long l3 = lmn3[0], m3 = lmn3[1], n3 = lmn3[2]
        long l4 = lmn4[0], m4 = lmn4[1], n4 = lmn4[2]
        double p = a+b
        double q = c+d
        double alpha = p*q/(p+q)
        double Px = (a*A[0] + b*B[0])/p
        double Py = (a*A[1] + b*B[1])/p
        double Pz = (a*A[2] + b*B[2])/p
        double Qx = (c*C[0] + d*D[0])/q
        double Qy = (c*C[1] + d*D[1])/q
        double Qz = (c*C[2] + d*D[2])/q
        double RPQ = sqrt(pow(Px-Qx,2) + \
                           pow(Py-Qy,2) + \
                           pow(Pz-Qz,2)) 

        long t,u,v,tau,nu,phi
        double val = 0.0
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
                                       alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 

    val *= 2*pow(pi,2.5)/(p*q*sqrt(p+q)) 
    return val 

