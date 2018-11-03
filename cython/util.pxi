import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport hyp1f1 

cdef double pi = 3.141592653589793238462643383279


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double E(int i,int j,int t,double Qx,double a,double b, int n = 0, double Ax = 0.0):
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
cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

def gaussian_product_center(double a, A, double b, B):
    return (a*np.asarray(A)+b*np.asarray(B))/(a+b)
    


