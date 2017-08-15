import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from mmd.molecule import * 
from mmd.realtime import * 
from mpl_toolkits.mplot3d import Axes3D
from mmd.utils.spectrum import * 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import scipy.interpolate
from matplotlib import cm
import matplotlib.gridspec as gridspec
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.contour import QuadContourSet

Swt  = np.load('swt.npy')
tsig = np.load('tsig.npy')
Sww  = np.load('sww.npy')
freq = np.load('freq.npy')

def compute_and_plot(ax,phase1):
    newS = np.zeros_like(Sww)
    for j in range(len(freq)):
        newS[j,:] = Sww[j,:]*np.exp(1j*phase1)
    CS = QuadContourSet(ax,freq*27.2114,freq*27.2114,np.abs(newS),10,cmap=cm.coolwarm,extend='both',filled=True)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(bottom=0.20)
compute_and_plot(ax,0.0)

axphase1 = plt.axes([0.25, 0.05, 0.65, 0.03])
sphase1 = Slider(axphase1,'$\Theta_1$', -np.pi,np.pi,valinit=0.0)

def update(ax,val):
    phase1 = sphase1.val
    ax.cla()
    compute_and_plot(ax,phase1)
    plt.draw()

sphase1.on_changed(lambda val: update(ax, val))

plt.show()
plt.close()


#
#plt.contourf(freq*27.2114,freq*27.2114,np.imag(Sww),30,cmap=cm.coolwarm,extend='both')
#plt.title('Imag')
#plt.xlabel('$\omega_1$')
#plt.ylabel('$\omega_3$')
#plt.show()
#plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(freq,freq)
surf = ax.plot_surface(X*27.2114,Y*27.2114,np.abs(Sww),cmap=cm.coolwarm)
plt.show()
