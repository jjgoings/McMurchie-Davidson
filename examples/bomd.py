from mmd.molecule import Molecule
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
dt = 5     # time step
steps = 100  # number steps

# saved lists for plotting data
X = []
Y = []
Z = []
E = []

# main BOMD loop
for _ in tqdm(range(steps)):
    # update positions
    for atom in mol.atoms:
        for q in range(3):
            atom.origin[q] += atom.velocities[q]*dt + 0.5*dt*dt*atom.forces[q]/atom.mass
            # save forces at t before update to forces at t + dt
            atom.saved_forces[q] = atom.forces[q]
    # update forces in lieu of updated nuclear positions
    mol.formBasis()
    mol.build()
    mol.RHF(doPrint=False)
    mol.forces()
    # update velocities
    for atom in mol.atoms:
        for q in range(3):
            atom.velocities[q] += 0.5*dt*(atom.saved_forces[q] + atom.forces[q])/atom.mass


    # append time-dependent positions and energies        
    X.append(abs(mol.atoms[0].origin[0] - mol.atoms[1].origin[0]))
    Y.append(abs(mol.atoms[0].origin[1] - mol.atoms[1].origin[1]))
    Z.append(abs(mol.atoms[0].origin[2] - mol.atoms[1].origin[2]))
    E.append(mol.energy)

# At completion, plot the energy as a function of time
plt.plot(np.arange(steps)*dt*0.02418884254,np.asarray(Z)) # convert time to fs
plt.xlabel('Time (fs)')
plt.ylabel('Bond length (bohr)')
plt.show()





