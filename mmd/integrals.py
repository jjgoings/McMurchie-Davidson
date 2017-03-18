from __future__ import division
import data 
import numpy as np
#import pyximport; pyximport.install()
from scipy.special import gamma, gammainc
from scipy.special import hyp1f1

def S(a,b,n=(0,0,0),gOrigin=np.zeros((3))):
    # Generalized overlap integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * S
    # normal overlap is just n = (0,0,0) case
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,n,gOrigin)
    return s

def Mu(a,b,C,direction):
    mu = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            mu += a.norm[ia]*b.norm[ib]*ca*cb*\
                     dipole(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,direction)
    return mu

def RxDel(a,b,C,direction,london=False):
    l = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            l += a.norm[ia]*b.norm[ib]*ca*cb*\
                     angular(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,direction,london)
    return l

def T(a,b,n=(0,0,0),gOrigin=np.zeros((3))):
    # Generalized kinetic integrals for derivatives of GIAOs
    # for basis function a centered at (Ax, Ay, Az)
    # n = (nx,ny,nz) for x_A^nx * y_A^ny * z_A^nz * del^2
    # normal kinetic is just n = (0,0,0) case
    t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin,n,gOrigin)
    return t

def V(a,b,C,n=(0,0,0),gOrigin=np.zeros((3))):
    v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,n,gOrigin)
    return v

#def ERI(a,b,c,d,n1=(0,0,0),n2=(0,0,0)):
#    eri = 0.0
#    for ja, ca in enumerate(a.coefs):
#        for jb, cb in enumerate(b.coefs):
#            for jc, cc in enumerate(c.coefs):
#                for jd, cd in enumerate(d.coefs):
#                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
#                             ca*cb*cc*cd*\
#                             electron_repulsion(a.exps[ja],a.shell,a.origin,\
#                                                b.exps[jb],b.shell,b.origin,\
#                                                c.exps[jc],c.shell,c.origin,\
#                                                d.exps[jd],d.shell,d.origin,\
#                                                n1,n2)
#    return eri

def overlap(a,lmn1,A,b,lmn2,B,n=(0,0,0),gOrigin=np.zeros((3))):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    S1 = E(l1,l2,0,A[0]-B[0],a,b,n[0],A[0]-gOrigin[0])
    S2 = E(m1,m2,0,A[1]-B[1],a,b,n[1],A[1]-gOrigin[1])
    S3 = E(n1,n2,0,A[2]-B[2],a,b,n[2],A[2]-gOrigin[2])
    return S1*S2*S3*np.power(np.pi/(a+b),1.5)

def dipole(a,lmn1,A,b,lmn2,B,C,direction):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    P = gaussian_product_center(a,A,b,B)
    if direction.lower() == 'x':
        XPC = P[0] - C[0]
        # Top call for 'D; works for sure, bottom works in terms of properties,
        # but the gauge-origin is different so the AO ints differ.
        #D  = E(l1,l2,1,A[0]-B[0],a,b) + XPC*E(l1,l2,0,A[0]-B[0],a,b)
        D  = E(l1,l2,0,A[0]-B[0],a,b,1,A[0]-C[0])
        S2 = E(m1,m2,0,A[1]-B[1],a,b)
        S3 = E(n1,n2,0,A[2]-B[2],a,b)
        return D*S2*S3*np.power(np.pi/(a+b),1.5)
    elif direction.lower() == 'y':
        YPC = P[1] - C[1]
        S1 = E(l1,l2,0,A[0]-B[0],a,b)
        #D  = E(m1,m2,1,A[1]-B[1],a,b) + YPC*E(m1,m2,0,A[1]-B[1],a,b)
        D  = E(m1,m2,0,A[1]-B[1],a,b,1,A[1]-C[1])
        S3 = E(n1,n2,0,A[2]-B[2],a,b)
        return S1*D*S3*np.power(np.pi/(a+b),1.5)
    elif direction.lower() == 'z':
        ZPC = P[2] - C[2]
        S1 = E(l1,l2,0,A[0]-B[0],a,b)
        S2 = E(m1,m2,0,A[1]-B[1],a,b)
        #D  = E(n1,n2,1,A[2]-B[2],a,b) + ZPC*E(n1,n2,0,A[2]-B[2],a,b)
        D  = E(n1,n2,0,A[2]-B[2],a,b,1,A[2]-C[2]) 
        return S1*S2*D*np.power(np.pi/(a+b),1.5)


''' OLD kinetic...cleaner but difficult to generalize for GIAO
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
'''
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

    return (Tx + Ty + Tz)*np.power(np.pi/(a+b),1.5)
          


def angular(a,lmn1,A,b,lmn2,B,C,direction,london):
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
        return -S0x*(S1y*D1z - S1z*D1y)*np.power(np.pi/(a+b),1.5) 

    elif direction.lower() == 'y':
        return -S0y*(S1z*D1x - S1x*D1z)*np.power(np.pi/(a+b),1.5) 

    elif direction.lower() == 'z':
        return -S0z*(S1x*D1y - S1y*D1x)*np.power(np.pi/(a+b),1.5) 


def E(i,j,t,Qx,a,b,n=0,Ax=0.0):
    p = a + b
    q = a*b/p
    if n == 0:
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
    else:
        return E(i+1,j,t,Qx,a,b,n-1,Ax) + Ax*E(i,j,t,Qx,a,b,n-1,Ax)

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

def nuclear_attraction(a,lmn1,A,b,lmn2,B,C,n,gOrigin=np.zeros((3))):
    # Generalized nucler integrals for derivatives of GIAOs
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
    val *= 2*np.pi/p # Pink book, Eq(9.9.40) 
    return val 

def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D,r1,r2):
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
    for t in xrange(l1+l2+1):
        for u in xrange(m1+m2+1):
            for v in xrange(n1+n2+1):
                for tau in xrange(l3+l4+1):
                    for nu in xrange(m3+m4+1):
                        for phi in xrange(n3+n4+1):
                            val += E(l1,l2,t,A[0]-B[0],a,b,r1[0],A[0]) * \
                                   E(m1,m2,u,A[1]-B[1],a,b,r1[1],A[1]) * \
                                   E(n1,n2,v,A[2]-B[2],a,b,r1[2],A[2]) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d,r2[0],C[0]) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d,r2[1],C[1]) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d,r2[2],C[2]) * \
                                   np.power(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ) 

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q)) 
    return val 

def boys(m,T):
    # pretty sure this works, tested a few cases vs wolfram alpha
    #if abs(T) < 1e-12:
    #    return 1/(2*m + 1)
    #else:
    #    return gammainc(m+0.5,T)*gamma(m+0.5)/(2*np.power(T,m+0.5))
    return hyp1f1(m+0.5,m+1.5,-T)/(2*m+1)

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)


