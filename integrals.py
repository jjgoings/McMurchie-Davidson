from __future__ import division
import data 
import numpy as np
from scipy.special import gamma, gammainc

def S(a,b):
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin)
    return s

def T(a,b):
    t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin)
    return t

def V(a,b,C):
    v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C)
    return v

def ERI(a,b,c,d):
    eri = 0.0
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

def overlap(a,lmn1,A,b,lmn2,B):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    S1 = E(l1,l2,0,A[0]-B[0],a,b)
    S2 = E(m1,m2,0,A[1]-B[1],a,b)
    S3 = E(n1,n2,0,A[2]-B[2],a,b)
    return S1*S2*S3*np.power(np.pi/(a+b),1.5)

def kinetic(a,lmn1,A,b,lmn2,B):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
                           (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
                  m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
                  n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2

def E(i,j,t,Qx,a,b):
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        return 0.0
    elif i == j == t == 0:
        return np.exp(-q*Qx*Qx)
    elif j == 0:
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - (q*Qx/a)*E(i-1,j,t,Qx,a,b) + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + (q*Qx/b)*E(i,j-1,t,Qx,a,b) + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)

def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
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

def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B)
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    val *= 2*np.pi/p # Pink book, Eq(9.9.40) 
    return val 

def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b
    q = c+d
    alpha = p*q/(p+q)
    P = gaussian_product_center(a,A,b,B)
    Q = gaussian_product_center(c,C,d,D)
    RPQ = np.linalg.norm(P-Q)

    val = 0.0
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
                                   np.power(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ) 

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q)) 
    return val 

def boys(m,T):
    # pretty sure this works, tested a few cases vs wolfram alpha
    if abs(T) < 1e-12:
        return 1/(2*m + 1)
    else:
        return gammainc(m+0.5,T)*gamma(m+0.5)/(2*np.power(T,m+0.5))

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)


