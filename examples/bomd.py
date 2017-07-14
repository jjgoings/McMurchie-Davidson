from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Simple example of Born-Oppenheimer Molecular Dynamics
# using minimal basis H2. Shows how you can chain together
# routines to create more complex programs. This runs about
# a a half of a femtosecond of molecular dynamics and plots
# energy as a function of time

# Molecular geometry input
h2 = """
0 1
H 0.0 0.0 0.74
H 0.0 0.0 0.00
"""

# init molecule and build integrals
mol = Molecule(geometry=h2,basis='sto-3g')
mol.build()

# do the SCF, compute initial forces on the atoms
mol.RHF()
mol.forces()

# BOMD parameters
dt = 0.1     # time step
steps = 200  # number steps

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
    mol.RHF(doPrint=False)
    mol.forces()
    # update velocities
    for atom in mol.atoms:
        for q in xrange(3):
            atom.velocities[q] += 0.5*dt*(atom.mass*atom.saved_forces[q] + atom.mass*atom.forces[q]) 


    # append time-dependent positions and energies        
    X.append(abs(mol.atoms[0].origin[0] - mol.atoms[1].origin[0]))
    Y.append(abs(mol.atoms[0].origin[1] - mol.atoms[1].origin[1]))
    Z.append(abs(mol.atoms[0].origin[2] - mol.atoms[1].origin[2]))
    E.append(mol.energy)

# At completion, plot the energy as a function of time
plt.plot(np.arange(steps)*dt,np.asarray(E))
plt.show()






