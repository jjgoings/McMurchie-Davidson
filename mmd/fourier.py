import numpy as np
from scipy.integrate import quad,nquad
import matplotlib.pyplot as plt

x = 1.0
y = 1.0
z = 1.0


def integrand(r,theta,phi,x,y,z):
    return np.sin(theta)*np.cos(x*r*np.sin(theta)*np.cos(phi) + y*r*np.sin(theta)*np.sin(phi) + z*r*np.cos(theta))

val = nquad(integrand,[[0,np.inf],[0,np.pi],[0,2*np.pi]],args=[x,y,z])

print val

