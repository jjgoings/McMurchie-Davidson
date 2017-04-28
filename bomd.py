from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# read in geometry
geometry = './geoms/h2.dat'

# init molecule and build integrals
mol = Molecule(filename=geometry,basis='sto-3g')
mol.build()
# do the SCF, compute initial forces on the atoms
scf = SCF(mol)
scf.RHF()
scf.forces()

# BOMD parameters
dt = 0.1
steps = 200

# saved lists for plotting data
X = []
Y = []
Z = []
E = []

# main BOMD loop
for step in tqdm(xrange(steps)):
    # update positions
    for atom in mol.atoms:
        for q in xrange(3):
            atom.origin[q] += atom.velocities[q]*dt + 0.5*dt*dt*atom.mass*atom.forces[q]
            # save forces at t before update to forces at t + dt
            atom.saved_forces[q] = atom.forces[q]
    # update forces in lieu of updated nuclear positions
    mol.formBasis()
    mol.build()
    scf.RHF(doPrint=False)
    scf.forces()
    # update velocities
    for atom in mol.atoms:
        for q in xrange(3):
            atom.velocities[q] += 0.5*dt*(atom.mass*atom.saved_forces[q] + atom.mass*atom.forces[q]) 


    # append time-dependent positions and energies        
    X.append(abs(mol.atoms[0].origin[0] - mol.atoms[1].origin[0]))
    Y.append(abs(mol.atoms[0].origin[1] - mol.atoms[1].origin[1]))
    Z.append(abs(mol.atoms[0].origin[2] - mol.atoms[1].origin[2]))
    E.append(mol.energy)

plt.plot(np.arange(steps)*dt,np.asarray(E))
plt.show()






