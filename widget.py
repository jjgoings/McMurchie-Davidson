import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mmd.utils.spectrum import *



times = np.load('timedata.npy')


dipole = times[0]
signal,freq = pade(np.arange(0,len(dipole))*0.1,dipole - dipole[0])
signal = -signal*freq

dipole = times[50]
signal2,freq = pade(np.arange(0,len(dipole))*0.1,dipole - dipole[0])
signal2 = -signal2*freq

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
omega = freq
s = signal 
s2 = signal2 
l, = plt.plot(omega*27.2114, s, lw=2, color='red')
m, = plt.plot(omega*27.2114, s2, lw=2, color='blue')
plt.axis([0, 30, -0.05, 0.2])

axcolor = 'lightgoldenrodyellow'
axphase = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

sphase = Slider(axphase, 'Phase', -np.pi, np.pi, valinit=0.0,valfmt='%1.4f')


def update(val):
    phase = sphase.val
    l.set_ydata(np.imag(signal*np.exp(1j*phase)))
    m.set_ydata(np.imag(signal2*np.exp(1j*phase)))
    fig.canvas.draw_idle()
sphase.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sphase.reset()
button.on_clicked(reset)


plt.show()
