import cython                                                                    
import numpy as np                                                               
cimport numpy as np                                                              
from libc.math cimport exp, pow, tgamma, sqrt, abs  

cdef double pi = 3.1415926535897932384626433832795028841971

cdef double G(double r,double a,int l1,double A):
    cdef double rA, rAsq
    rA   = pow(r - A,l1)
    rAsq = pow(r - A,2)
    return rA*exp(-a*rAsq)

cdef double wG(double r,double a,int l1,double A):
    return pow(r - A,l1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double eri_integrand(np.ndarray[double, mode='c', ndim=1] x,double a, int l1, int m1, int n1,double Ax,double Ay, double Az, double b,int l2, int m2, int n2, double Bx, double By, double Bz,double c,int l3, int m3, int n3, double Cx,double Cy,double Cz,double d,int l4, int m4, int n4,double Dx, double Dy, double Dz,int nx1,int ny1, int nz1, int nx2, int ny2, int nz2):
    cdef double x1 = x[0]
    cdef double y1 = x[1]
    cdef double z1 = x[2]
    cdef double x2 = x[3]
    cdef double y2 = x[4]
    cdef double z2 = x[5]
    return pow(x1,nx1)*G(x1,a,l1,Ax) * \
           pow(y1,ny1)*G(y1,a,m1,Ay) * \
           pow(z1,nz1)*G(z1,a,n1,Az) * \
           G(x1,b,l2,Bx) * \
           G(y1,b,m2,By) * \
           G(z1,b,n2,Bz) * \
           1.0/sqrt(pow(x1 - x2,2) + pow(y1-y2,2) + pow(z1-z2,2)) * \
           pow(x2,nx2)*G(x2,c,l3,Cx) * \
           pow(y2,ny2)*G(y2,c,m3,Cy) * \
           pow(z2,nz2)*G(z2,c,n3,Cz) * \
           G(x2,d,l4,Dx) * \
           G(y2,d,m4,Dy) * \
           G(z2,d,n4,Dz) 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double eri_wintegrand(np.ndarray[double, ndim=1] x,double a, int l1, int m1, int n1,double Ax,double Ay, double Az, double b,int l2, int m2, int n2, double Bx, double By, double Bz,double c,int l3, int m3, int n3, double Cx,double Cy,double Cz,double d,int l4, int m4, int n4,double Dx, double Dy, double Dz,int nx1,int ny1, int nz1, int nx2, int ny2, int nz2, double gx, double gy, double gz):
    cdef double x1 = x[0]
    cdef double y1 = x[1]
    cdef double z1 = x[2]
    cdef double x2 = x[3]
    cdef double y2 = x[4]
    cdef double z2 = x[5]
    cdef double preAB = sqrt(pi/(a+b))
    cdef double gammaAB = a*b/(a+b)
    cdef double preCD = sqrt(pi/(c+d))
    cdef double gammaCD = c*d/(c+d)
    cdef double pre1x = preAB*exp(-gammaAB*pow(Ax-Bx,2))
    cdef double pre1y = preAB*exp(-gammaAB*pow(Ay-By,2))
    cdef double pre1z = preAB*exp(-gammaAB*pow(Az-Bz,2))
    cdef double pre2x = preCD*exp(-gammaCD*pow(Cx-Dx,2))
    cdef double pre2y = preCD*exp(-gammaCD*pow(Cy-Dy,2))
    cdef double pre2z = preCD*exp(-gammaCD*pow(Cz-Dz,2))
    return pow(x1 - gx,nx1)*wG(x1,a,l1,Ax) * \
           pow(y1 - gy,ny1)*wG(y1,a,m1,Ay) * \
           pow(z1 - gz,nz1)*wG(z1,a,n1,Az) * \
           wG(x1,b,l2,Bx) * \
           wG(y1,b,m2,By) * \
           wG(z1,b,n2,Bz) * \
           1.0/sqrt(pow(x1 - x2,2) + pow(y1-y2,2) + pow(z1-z2,2)) * \
           pow(x2 - gx,nx2)*wG(x2,c,l3,Cx) * \
           pow(y2 - gy,ny2)*wG(y2,c,m3,Cy) * \
           pow(z2 - gz,nz2)*wG(z2,c,n3,Cz) * \
           wG(x2,d,l4,Dx) * \
           wG(y2,d,m4,Dy) * \
           wG(z2,d,n4,Dz) * pre1x * pre1y * pre1z * pre2x *pre2y * pre2z 



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=2] distribution(int size, double sp, double Px, double Py, double Pz, double sq,double Qx, double Qy, double Qz):
    cdef np.ndarray[double,ndim=1] x1 = np.random.normal(Px,sp,size=size)
    cdef np.ndarray[double,ndim=1] y1 = np.random.normal(Py,sp,size=size)
    cdef np.ndarray[double,ndim=1] z1 = np.random.normal(Pz,sp,size=size)
    cdef np.ndarray[double,ndim=1] x2 = np.random.normal(Qx,sq,size=size)
    cdef np.ndarray[double,ndim=1] y2 = np.random.normal(Qy,sq,size=size)
    cdef np.ndarray[double,ndim=1] z2 = np.random.normal(Qz,sq,size=size)
    return np.array((x1,y1,z1,x2,y2,z2)).T





