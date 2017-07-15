from __future__ import division
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
include "util.pxi"

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double [:,:,:,:] doERIs(long N,double [:,:,:,:] TwoE, object bfs):
    cdef long i,j,k,l,ij,kl
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
    #print "\n\n"

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double [:,:,:,:,:] do2eGIAO(long N,double [:,:,:,:,:] GR1, double [:,:,:,:,:] GR2, double [:,:,:,:,:] dgdb, object bfs,double [:] gauge_origin):
    cdef long ij = 0
    cdef long i,j,k,l,kl,ik
    cdef double XMN,YMN,ZMN,XPQ,YPQ,ZPQ
    cdef double GR1x,GR1y,GR1z,GR2x,GR2y,GR2z
    for i in (range(N)):
        for j in range(N):
            ij += 1
            kl = 0
            for k in range(N):
                ik = i + k
                for l in range(N):
                    kl += 1
                    if (ij >= kl and ik >= j+l and not (i==j and k==l)):
                        #QMN matrix elements
                        XMN  = bfs[i].origin[0] - bfs[j].origin[0]
                        YMN  = bfs[i].origin[1] - bfs[j].origin[1]
                        ZMN  = bfs[i].origin[2] - bfs[j].origin[2]
                        #QPQ matrix elements
                        XPQ  = bfs[k].origin[0] - bfs[l].origin[0]
                        YPQ  = bfs[k].origin[1] - bfs[l].origin[1]
                        ZPQ  = bfs[k].origin[2] - bfs[l].origin[2]
  
                        GR1x = ERI(bfs[i],bfs[j],bfs[k],bfs[l],n1=(1,0,0),n2=(0,0,0), gOrigin=gauge_origin)
                        GR1y = ERI(bfs[i],bfs[j],bfs[k],bfs[l],n1=(0,1,0),n2=(0,0,0), gOrigin=gauge_origin)
                        GR1z = ERI(bfs[i],bfs[j],bfs[k],bfs[l],n1=(0,0,1),n2=(0,0,0), gOrigin=gauge_origin)
                        GR2x = ERI(bfs[i],bfs[j],bfs[k],bfs[l],n1=(0,0,0),n2=(1,0,0), gOrigin=gauge_origin)
                        GR2y = ERI(bfs[i],bfs[j],bfs[k],bfs[l],n1=(0,0,0),n2=(0,1,0), gOrigin=gauge_origin)
                        GR2z = ERI(bfs[i],bfs[j],bfs[k],bfs[l],n1=(0,0,0),n2=(0,0,1), gOrigin=gauge_origin)
 
                        # add QMN contribution
                        GR1[0,i,j,k,l] = 0.5*(-ZMN*GR1y + YMN*GR1z)
                        GR1[1,i,j,k,l] = 0.5*( ZMN*GR1x - XMN*GR1z)
                        GR1[2,i,j,k,l] = 0.5*(-YMN*GR1x + XMN*GR1y)
                        # add QPQ contribution
                        GR2[0,i,j,k,l] = 0.5*(-ZPQ*GR2y + YPQ*GR2z)
                        GR2[1,i,j,k,l] = 0.5*( ZPQ*GR2x - XPQ*GR2z)
                        GR2[2,i,j,k,l] = 0.5*(-YPQ*GR2x + XPQ*GR2y)
  
                        dgdb[0,i,j,k,l] = dgdb[0,k,l,i,j] = GR1[0,i,j,k,l] + GR2[0,i,j,k,l]
                        dgdb[1,i,j,k,l] = dgdb[1,k,l,i,j] = GR1[1,i,j,k,l] + GR2[1,i,j,k,l]
                        dgdb[2,i,j,k,l] = dgdb[2,k,l,i,j] = GR1[2,i,j,k,l] + GR2[2,i,j,k,l]

                        dgdb[0,j,i,l,k] = dgdb[0,l,k,j,i] = -dgdb[0,i,j,k,l]
                        dgdb[1,j,i,l,k] = dgdb[1,l,k,j,i] = -dgdb[1,i,j,k,l]
                        dgdb[2,j,i,l,k] = dgdb[2,l,k,j,i] = -dgdb[2,i,j,k,l]
    return dgdb

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERI(object a,object b,object c,object d, tuple n1 = (0,0,0), tuple n2 = (0,0,0), gOrigin = np.zeros((3)) ):
    cdef double eri = 0.0
    cdef int ja, jb, jc, jd
    cdef double ca, cb, cc, cd
    cdef double [:] aExps = np.asarray(a.exps), bExps = np.asarray(b.exps), cExps = np.asarray(c.exps), dExps = np.asarray(d.exps)
    cdef double [:] aCoefs = np.asarray(a.coefs), bCoefs = np.asarray(b.coefs), cCoefs = np.asarray(c.coefs), dCoefs = np.asarray(d.coefs)
    cdef double [:] aNorm = np.asarray(a.norm), bNorm = np.asarray(b.norm), cNorm = np.asarray(c.norm), dNorm = np.asarray(d.norm)
    cdef long   [:] aShell = a.shell, bShell = b.shell, cShell = c.shell, dShell = d.shell
    cdef double [:] aOrigin = a.origin, bOrigin = b.origin, cOrigin = c.origin, dOrigin = d.origin
    cdef long A = len(a.coefs), B = len(b.coefs), C = len(c.coefs), D = len(d.coefs)
    cdef long [:] N1 = np.asarray(n1,dtype='int')
    cdef long [:] N2 = np.asarray(n2,dtype='int')
    cdef double [:] GO = gOrigin
    for ja in range(A):
        for jb in range(B):
            for jc in range(C):
                for jd in range(D):
                    eri += aNorm[ja]*bNorm[jb]*cNorm[jc]*dNorm[jd]*\
                             aCoefs[ja]*bCoefs[jb]*cCoefs[jc]*dCoefs[jd]*\
                             electron_repulsion(aExps[ja],aShell,aOrigin,\
                                                bExps[jb],bShell,bOrigin,\
                                                cExps[jc],cShell,cOrigin,\
                                                dExps[jd],dShell,dOrigin,\
                                                N1,N2, GO)
#   # for ja, ca in enumerate(a.coefs):
#   #     for jb, cb in enumerate(b.coefs):
#   #         for jc, cc in enumerate(c.coefs):
#   #             for jd, cd in enumerate(d.coefs):
#   #                 eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
#   #                          ca*cb*cc*cd*\
#   #                          electron_repulsion(a.exps[ja],a.shell,a.origin,\
#   #                                             b.exps[jb],b.shell,b.origin,\
#   #                                             c.exps[jc],c.shell,c.origin,\
#   #                                             d.exps[jd],d.shell,d.origin,\
#   #                                             np.asarray(n1,dtype='int'),np.asarray(n2,dtype='int'), gOrigin)
    return eri

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double electron_repulsion(double a, long [:] lmn1, double [:] A, double b, long [:] lmn2, double [:] B,double c, long [:] lmn3, double [:] C,double d, long [:] lmn4, double [:] D, long [:] r1, long [:] r2, double [:] gOrigin):
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

