from __future__ import division
#import data 
import cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport gammainc, hyp1f1 
from scipy.misc import factorial2 as fact2
from tqdm import tqdm

cdef double pi = 3.141592653589793238462643383279

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double S(object a, object b, tuple n=(0,0,0), double [:] gOrigin=np.zeros((3))):
    # Generalized overlap integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * S
    # normal overlap is just n = (0,0,0) case
    cdef double s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,n,gOrigin)
    return s

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double Sx(object a, object b, tuple n=(0,0,0), double [:] gOrigin=np.zeros((3)),int x = 0, str center = 'A'):
    # Generalized overlap derivative integrals 
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * S
    # normal overlap is just n = (0,0,0) case
    cdef double s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlapX(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,n,gOrigin,x,center)
    return s

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double Mu(object a, object b,str direction, tuple n=(0,0,0),double [:] gOrigin=np.zeros((3))):
    cdef double mu = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            mu += a.norm[ia]*b.norm[ib]*ca*cb*\
                     dipole(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,direction,n,gOrigin)
    return mu

def RxDel(a, b, C, direction, london=False):
    l = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            l += a.norm[ia]*b.norm[ib]*ca*cb*\
                     angular(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,direction,london)
    return l

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double T(object a, object b,tuple n=(0,0,0), double [:] gOrigin=np.zeros((3))):
    # Generalized kinetic integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * del^2
    # normal kinetic is just n = (0,0,0) case
    cdef double t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin,n,gOrigin)
    return t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double Tx(object a, object b,tuple n=(0,0,0), double [:] gOrigin=np.zeros((3)),int x = 0, str center = 'A'):
    # Generalized kinetic integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * del^2
    # normal kinetic is just n = (0,0,0) case
    cdef double t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kineticX(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin,n,gOrigin,x, center)
    return t


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double V(object a, object b, double [:] C, tuple n=(0,0,0), double [:] gOrigin=np.zeros((3))):
    # Generalized nuclear attraction integrals for derivatives of GIAOs
    # nucleus is centered at 'C'
    # normal nuclear attraction corresponse to n = (0,0,0) case
    cdef double v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,n,gOrigin)
    return v

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double VxA(object a, object b, double [:] C, tuple n=(0,0,0), double [:] gOrigin=np.zeros((3)),int x = 0):
    # handles operator derivative contribution, e.g. Hellman Feynman forces
    # Generalized nuclear attraction integrals for derivatives of GIAOs
    # nucleus is centered at 'C'
    # normal nuclear attraction corresponse to n = (0,0,0) case
    cdef double v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attractionXa(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,n,gOrigin,x)
    return v

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double VxB(object a, object b, np.ndarray C, tuple n=(0,0,0), double [:] gOrigin=np.zeros((3)),int x = 0, str center = 'A'):
    # handles overlap derivative contribution to nuclear attraction derivatives
    # Generalized nuclear attraction integrals for derivatives of GIAOs
    # nucleus is centered at 'C'
    # normal nuclear attraction corresponse to n = (0,0,0) case
    cdef double v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attractionXb(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,n,gOrigin,x,center)
    return v


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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERIx(object a,object b,object c,object d, tuple n1 = (0,0,0), tuple n2 = (0,0,0), gOrigin = np.zeros((3)), int x = 0, str center = 'a'):
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
                             electron_repulsionX(aExps[ja],aShell,aOrigin,\
                                                bExps[jb],bShell,bOrigin,\
                                                cExps[jc],cShell,cOrigin,\
                                                dExps[jd],dShell,dOrigin,\
                                                N1,N2, GO, x, center.lower())
    return eri


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double Ex(int i,int j,int t,double Qx,double a,double b, int n = 0, double Ax = 0.0, int q = 0, int r = 0) nogil:
    # only handling first derivatives
    if q == 1:
        return 2*a*E(i+1,j,t,Qx,a,b,n,Ax) - i*E(i-1,j,t,Qx,a,b,n,Ax)
    elif r == 1:
        return 2*b*E(i,j+1,t,Qx,a,b,n,Ax) - j*E(i,j-1,t,Qx,a,b,n,Ax)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double E(int i,int j,int t,double Qx,double a,double b, int n = 0, double Ax = 0.0) nogil:
    p = a + b
    u = a*b/p
    if n == 0:
        if (t < 0) or (t > (i + j)):
            return 0.0
        elif i == j == t == 0:
            return exp(-u*Qx*Qx)
        elif j == 0:
            return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - (u*Qx/a)*E(i-1,j,t,Qx,a,b) + \
                   (t+1)*E(i-1,j,t+1,Qx,a,b)
        else:
            return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + (u*Qx/b)*E(i,j-1,t,Qx,a,b) + \
                   (t+1)*E(i,j-1,t+1,Qx,a,b)
    else:
        return E(i+1,j,t,Qx,a,b,n-1,Ax) + Ax*E(i,j,t,Qx,a,b,n-1,Ax)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double R(int t,int u,int v,int n, double p,double PCx, double PCy, double PCz, double RPC) nogil:
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
cdef double electron_repulsion(double a, long [:] lmn1, double [:] A, double b, long [:] lmn2, double [:] B,double c, long [:] lmn3, double [:] C,double d, long [:] lmn4, double [:] D, long [:] r1, long [:] r2, double [:] gOrigin) nogil:
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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double electron_repulsionX(double a, long [:] lmn1, double [:] A, double b, long [:] lmn2, double [:] B,double c, long [:] lmn3, double [:] C,double d, long [:] lmn4, double [:] D, long [:] r1, long [:] r2, double [:] gOrigin, int x = 0, str center = 'a'):
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
    if center == 'a':
        if x == 0:
            val = 0.0
            for t in range(l1+l2+1+r1x+1):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += Ex(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0],q=1,r=0) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 1:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y+1):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           Ex(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1],q=1,r=0) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 2:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z+1):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           Ex(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2],q=1,r=0) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
    elif center == 'b':
        if x == 0:
            val = 0.0
            for t in range(l1+l2+1+r1x+1):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += Ex(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0],q=0,r=1) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 1:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y+1):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           Ex(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1],q=0,r=1) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 2:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z+1):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           Ex(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2],q=0,r=1) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 

    elif center == 'c':
        if x == 0:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x+1):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           Ex(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0],q=1,r=0) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 1:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y+1):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           Ex(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1],q=1,r=0) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 2:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z+1):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z+1):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           Ex(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2],q=1,r=0) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 

    elif center == 'd':
        if x == 0:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x+1):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           Ex(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0],q=0,r=1) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 1:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y+1):
                                for phi in range(n3+n4+1+r2z):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           Ex(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1],q=0,r=1) * \
                                           E(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2]) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        elif x == 2:
            val = 0.0
            for t in range(l1+l2+1+r1x):
                for u in range(m1+m2+1+r1y):
                    for v in range(n1+n2+1+r1z+1):
                        for tau in range(l3+l4+1+r2x):
                            for nu in range(m3+m4+1+r2y):
                                for phi in range(n3+n4+1+r2z+1):
                                    val += E(l1,l2,t,A[0]-B[0],a,b,r1x,A[0] - gOrigin[0]) * \
                                           E(m1,m2,u,A[1]-B[1],a,b,r1y,A[1] - gOrigin[1]) * \
                                           E(n1,n2,v,A[2]-B[2],a,b,r1z,A[2] - gOrigin[2]) * \
                                           E(l3,l4,tau,C[0]-D[0],c,d,r2x,C[0] - gOrigin[0]) * \
                                           E(m3,m4,nu ,C[1]-D[1],c,d,r2y,C[1] - gOrigin[1]) * \
                                           Ex(n3,n4,phi,C[2]-D[2],c,d,r2z,C[2] - gOrigin[2],q=0,r=1) * \
                                           pow(-1,tau+nu+phi) * \
                                           R(t+tau,u+nu,v+phi,0,\
                                               alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 
        
    val *= 2*pow(pi,2.5)/(p*q*sqrt(p+q)) 
    return val 

@cython.cdivision(True)
cdef double boys(double m,double T) nogil:
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

def overlap(a,lmn1,A,b,lmn2,B,n=(0,0,0),gOrigin=np.zeros((3))):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
    return S1*S2*S3*np.power(pi/(a+b),1.5)

def overlapX(a,lmn1,A,b,lmn2,B,n=(0,0,0),gOrigin=np.zeros((3)),x=0,center='A'):
    # can only handle first derivatives
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    if center.lower() == 'a':      
        if x == 0:
            S1 = Ex(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0)
            S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
            S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
        elif x == 1:
            S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
            S2 = Ex(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0)
            S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
        elif x == 2:
            S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
            S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
            S3 = Ex(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0)
        else:
            # to appease the Cython
            S1 = 0.0
            S2 = 0.0
            S3 = 0.0

    elif center.lower() == 'b':      
        if x == 0:
            S1 = Ex(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1)
            S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
            S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
        elif x == 1:
            S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
            S2 = Ex(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1)
            S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
        elif x == 2:
            S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
            S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
            S3 = Ex(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1)
        else:
            # to appease the Cython
            S1 = 0.0
            S2 = 0.0
            S3 = 0.0
    else:
        # to appease the Cython
        S1 = 0.0
        S2 = 0.0
        S3 = 0.0
    return S1*S2*S3*np.power(pi/(a+b),1.5)


def dipole(a,lmn1,A,b,lmn2,B,direction,n=(0,0,0),gOrigin=np.zeros((3))):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    P = gaussian_product_center(a,A,b,B)
    if direction.lower() == 'x':
        #XPC = P[0] - C[0]
        # Top call for 'D; works for sure, bottom works in terms of properties,
        # but the gauge-origin is different so the AO ints differ.
        #D  = E(l1,l2,1,A[0]-B[0],a,b) + XPC*E(l1,l2,0,A[0]-B[0],a,b)
        D  = E(l1,l2,0,A[0]-B[0],a,b,1+n[0],A[0]-gOrigin[0])
        S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
        S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2] -gOrigin[2])
        return D*S2*S3*np.power(pi/(a+b),1.5)
    elif direction.lower() == 'y':
        #YPC = P[1] - C[1]
        S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
        #D  = E(m1,m2,1,A[1]-B[1],a,b) + YPC*E(m1,m2,0,A[1]-B[1],a,b)
        D  = E(m1,m2,0,A[1]-B[1],a,b,1+n[1],A[1]-gOrigin[1])
        S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
        return S1*D*S3*np.power(pi/(a+b),1.5)
    elif direction.lower() == 'z':
        #ZPC = P[2] - C[2]
        S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
        S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
        #D  = E(n1,n2,1,A[2]-B[2],a,b) + ZPC*E(n1,n2,0,A[2]-B[2],a,b)
        D  = E(n1,n2,0,A[2]-B[2],a,b,1+n[2],A[2]-gOrigin[2]) 
        return S1*S2*D*np.power(pi/(a+b),1.5)

def kinetic(a,lmn1,A,b,lmn2,B,n=(0,0,0),gOrigin=np.zeros((3))):
    # explicit kinetic in terms of "E" operator
    # generalized to include GIAO derivatives
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    Ax,Ay,Az = (2*np.asarray(lmn2) + 1)*b
    Bx = By = Bz = -2*np.power(b,2) # redundant, I know
    Cx,Cy,Cz = -0.5*np.asarray(lmn2)*(np.asarray(lmn2)-1) 

    Tx = Ax*E(l1,l2  ,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) + \
         Bx*E(l1,l2+2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) + \
         Cx*E(l1,l2-2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    Tx *= E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    Tx *= E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])

    Ty = Ay*E(m1,m2  ,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) + \
         By*E(m1,m2+2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) + \
         Cy*E(m1,m2-2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    Ty *= E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    Ty *= E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])

    Tz = Az*E(n1,n2  ,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) + \
         Bz*E(n1,n2+2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) + \
         Cz*E(n1,n2-2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
    Tz *= E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    Tz *= E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])

    return (Tx + Ty + Tz)*np.power(pi/(a+b),1.5)
          
def kineticX(a,lmn1,A,b,lmn2,B,n=(0,0,0),gOrigin=np.zeros((3)),x=0, center = 'A'):
    # explicit kinetic in terms of "E" operator
    # generalized to include GIAO derivatives
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    Ax,Ay,Az = (2*np.asarray(lmn2) + 1)*b
    Bx = By = Bz = -2*np.power(b,2) # redundant, I know
    Cx,Cy,Cz = -0.5*np.asarray(lmn2)*(np.asarray(lmn2)-1) 

    if center.lower() == 'a':
        if x == 0:
            Tx = Ax*Ex(l1,l2  ,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0) + \
                 Bx*Ex(l1,l2+2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0) + \
                 Cx*Ex(l1,l2-2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0)
        else:
            Tx = Ax*E(l1,l2  ,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) + \
                 Bx*E(l1,l2+2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) + \
                 Cx*E(l1,l2-2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    
        if x == 1:
            Ty = Ay*Ex(m1,m2  ,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0) + \
                 By*Ex(m1,m2+2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0) + \
                 Cy*Ex(m1,m2-2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0)
        else:
            Ty = Ay*E(m1,m2  ,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) + \
                 By*E(m1,m2+2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) + \
                 Cy*E(m1,m2-2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    
        if x == 2:
            Tz = Az*Ex(n1,n2  ,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0) + \
                 Bz*Ex(n1,n2+2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0) + \
                 Cz*Ex(n1,n2-2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0)
        else:
            Tz = Az*E(n1,n2  ,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) + \
                 Bz*E(n1,n2+2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) + \
                 Cz*E(n1,n2-2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
    
        if x == 0:
            Ty *= Ex(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0)
            Tz *= Ex(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0)
        else:
            Ty *= E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
            Tz *= E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    
        if x == 1:
            Tx *= Ex(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0)
            Tz *= Ex(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0)
        else:
            Tx *= E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
            Tz *= E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    
        if x == 2:
            Tx *= Ex(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0)
            Ty *= Ex(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0)
        else:
            Tx *= E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
            Ty *= E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])

    if center.lower() == 'b':
        if x == 0:
            Tx = Ax*Ex(l1,l2  ,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1) + \
                 Bx*Ex(l1,l2+2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1) + \
                 Cx*Ex(l1,l2-2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1)
        else:
            Tx = Ax*E(l1,l2  ,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) + \
                 Bx*E(l1,l2+2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) + \
                 Cx*E(l1,l2-2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    
        if x == 1:
            Ty = Ay*Ex(m1,m2  ,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1) + \
                 By*Ex(m1,m2+2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1) + \
                 Cy*Ex(m1,m2-2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1)
        else:
            Ty = Ay*E(m1,m2  ,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) + \
                 By*E(m1,m2+2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) + \
                 Cy*E(m1,m2-2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    
        if x == 2:
            Tz = Az*Ex(n1,n2  ,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1) + \
                 Bz*Ex(n1,n2+2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1) + \
                 Cz*Ex(n1,n2-2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1)
        else:
            Tz = Az*E(n1,n2  ,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) + \
                 Bz*E(n1,n2+2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) + \
                 Cz*E(n1,n2-2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
    
        if x == 0:
            Ty *= Ex(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1)
            Tz *= Ex(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1)
        else:
            Ty *= E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
            Tz *= E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    
        if x == 1:
            Tx *= Ex(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1)
            Tz *= Ex(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1)
        else:
            Tx *= E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
            Tz *= E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    
        if x == 2:
            Tx *= Ex(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1)
            Ty *= Ex(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1)
        else:
            Tx *= E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
            Ty *= E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
    
    return (Tx + Ty + Tz)*np.power(pi/(a+b),1.5)
          


def angular(a, lmn1, A, b, lmn2, B, C, direction,london):
    # a little extra work at the moment, but not all that more expensive
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    P = gaussian_product_center(a,A,b,B)

    XPC = P[0] - C[0]
    YPC = P[1] - C[1]
    ZPC = P[2] - C[2]

    S0x =    E(l1,l2,0,A[0]-B[0],a,b) 
    S0y =    E(m1,m2,0,A[1]-B[1],a,b) 
    S0z =    E(n1,n2,0,A[2]-B[2],a,b) 

    if london:
        # pretty sure this works
        S1x = E(l1,l2,0,A[0]-B[0],a,b,1,A[0]-B[0])
        S1y = E(m1,m2,0,A[1]-B[1],a,b,1,A[1]-B[1])
        S1z = E(n1,n2,0,A[2]-B[2],a,b,1,A[2]-B[2])
        #S1x = E(l1,l2,0,A[0]-B[0],a,b,1,A[0])
        #S1y = E(m1,m2,0,A[1]-B[1],a,b,1,A[1])
        #S1z = E(n1,n2,0,A[2]-B[2],a,b,1,A[2])
    else:
        # old code, works
        #S1x = E(l1,l2,1,A[0]-B[0],a,b) + XPC*E(l1,l2,0,A[0]-B[0],a,b)
        #S1y = E(m1,m2,1,A[1]-B[1],a,b) + YPC*E(m1,m2,0,A[1]-B[1],a,b)
        #S1z = E(n1,n2,1,A[2]-B[2],a,b) + ZPC*E(n1,n2,0,A[2]-B[2],a,b)
        S1x = E(l1,l2,0,A[0]-B[0],a,b,1,A[0]-C[0])
        S1y = E(m1,m2,0,A[1]-B[1],a,b,1,A[1]-C[1])
        S1z = E(n1,n2,0,A[2]-B[2],a,b,1,A[2]-C[2])
    

    D1x = l2*E(l1,l2-1,0,A[0]-B[0],a,b) - 2*b*E(l1,l2+1,0,A[0]-B[0],a,b)
    D1y = m2*E(m1,m2-1,0,A[1]-B[1],a,b) - 2*b*E(m1,m2+1,0,A[1]-B[1],a,b)
    D1z = n2*E(n1,n2-1,0,A[2]-B[2],a,b) - 2*b*E(n1,n2+1,0,A[2]-B[2],a,b)

    if direction.lower() == 'x':
        return -S0x*(S1y*D1z - S1z*D1y)*np.power(pi/(a+b),1.5) 

    elif direction.lower() == 'y':
        return -S0y*(S1z*D1x - S1x*D1z)*np.power(pi/(a+b),1.5) 

    elif direction.lower() == 'z':
        return -S0z*(S1x*D1y - S1y*D1x)*np.power(pi/(a+b),1.5) 

def nuclear_attraction(a,lmn1,A,b,lmn2,B,C,n,gOrigin=np.zeros((3))):
    # Generalized nuclear integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * 1/r
    # normal nuclear attraction is just n = (0,0,0) case
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B)
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in xrange(l1+l2+1+n[0]):
        for u in xrange(m1+m2+1+n[1]):
            for v in xrange(n1+n2+1+n[2]):
                val += E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                       E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                       E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    val *= 2*pi/p # Pink book, Eq(9.9.40) 
    return val 

def nuclear_attractionXa(a,lmn1,A,b,lmn2,B,C,n,gOrigin=np.zeros((3)),x = 0):
    # First part: compute V_ab^(1,0,0) -like terms
    # Generalized nuclear integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * 1/r
    # normal nuclear attraction is just n = (0,0,0) case
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B)
    RPC = np.linalg.norm(P-C)

    # FIXME need to restructure
    # see Eq(204) in Helgaker, Modern Elec. Struc Theory (p820)
    if x == 0:
        val = 0.0
        for t in xrange(l1+l2+1+n[0]):
            for u in xrange(m1+m2+1+n[1]):
                for v in xrange(n1+n2+1+n[2]):
                    val -= E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                           E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                           E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                           R(t+1,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    elif x == 1:
        val = 0.0
        for t in xrange(l1+l2+1+n[0]):
            for u in xrange(m1+m2+1+n[1]):
                for v in xrange(n1+n2+1+n[2]):
                    val -= E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                           E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                           E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                           R(t,u+1,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    elif x == 2:
        val = 0.0
        for t in xrange(l1+l2+1+n[0]):
            for u in xrange(m1+m2+1+n[1]):
                for v in xrange(n1+n2+1+n[2]):
                    val -= E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                           E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                           E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                           R(t,u,v+1,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    val *= 2*pi/p # Pink book, Eq(9.9.40) 
    return val 

def nuclear_attractionXb(a, lmn1, A, b, lmn2, B, C,  n, gOrigin=np.zeros((3)), x = 0, center = 'A'):
    # Second part: compute d/dX V_ab^(0,0,0) like terms
    # Generalized nuclear integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * 1/r
    # normal nuclear attraction is just n = (0,0,0) case
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    val = 0.0 
    P = gaussian_product_center(a,A,b,B)
    RPC = np.linalg.norm(P-C)

    # FIXME need to restructure
    # see Eq(204) in Helgaker, Modern Elec. Struc Theory (p820)
    # note that t,u,v must be incremented by one wrt correct derivative
    # for example, compare with regular nuclear attraction
    if center.lower() == 'a':
        if x == 0:
            val = 0.0
            for t in xrange(l1+l2+1+n[0]+1):
                for u in xrange(m1+m2+1+n[1]):
                    for v in xrange(n1+n2+1+n[2]):
                        val += Ex(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=1,r=0) * \
                               E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                               E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                               R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
        elif x == 1:
            val = 0.0
            for t in xrange(l1+l2+1+n[0]):
                for u in xrange(m1+m2+1+n[1]+1):
                    for v in xrange(n1+n2+1+n[2]):
                        val += E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                               Ex(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=1,r=0) * \
                               E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                               R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
        elif x == 2:
            val = 0.0
            for t in xrange(l1+l2+1+n[0]):
                for u in xrange(m1+m2+1+n[1]):
                    for v in xrange(n1+n2+1+n[2]+1):
                        val += E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                               E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                               Ex(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=1,r=0) * \
                               R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    if center.lower() == 'b':
        if x == 0:
            val = 0.0
            for t in xrange(l1+l2+1+n[0]+1):
                for u in xrange(m1+m2+1+n[1]):
                    for v in xrange(n1+n2+1+n[2]):
                        val += Ex(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0],q=0,r=1) * \
                               E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                               E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                               R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
        elif x == 1:
            val = 0.0
            for t in xrange(l1+l2+1+n[0]):
                for u in xrange(m1+m2+1+n[1]+1):
                    for v in xrange(n1+n2+1+n[2]):
                        val += E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                               Ex(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1],q=0,r=1) * \
                               E(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2]) * \
                               R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
        elif x == 2:
            val = 0.0
            for t in xrange(l1+l2+1+n[0]):
                for u in xrange(m1+m2+1+n[1]):
                    for v in xrange(n1+n2+1+n[2]+1):
                        val += E(l1,l2,t,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0]) * \
                               E(m1,m2,u,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1]) * \
                               Ex(n1,n2,v,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2],q=0,r=1) * \
                               R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    val *= 2*pi/p # Pink book, Eq(9.9.40) 
    return val 


def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)


