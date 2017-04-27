from mmd.molecule import * 
from mmd.scf import * 
from mmd.postscf import * 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(suppress=True,precision=6)

# read in geometry
geometry = './geoms/h2.dat'

# init molecule and build integrals
mol = Molecule(filename=geometry,basis='sto-3g')
mol.build()
# do the SCF
scf = SCF(mol)
scf.RHF()
scf.forces()

dt = 0.05
steps = 200
X = []
Y = []
Z = []
for step in tqdm(xrange(steps)):
    atom_old = mol.atoms
    # update positions
    for atom in mol.atoms:
        for q in xrange(3):
            atom.origin[q] += atom.velocities[q]*dt + 0.5*dt*dt*atom.mass*atom.forces[q]
    # update forces
    scf.RHF(doPrint=False)
    scf.forces()
    # update velocities
    for idx, atom in enumerate(mol.atoms):
        for q in xrange(3):
            atom.velocities[q] += 0.5*dt*(atom.mass*atom_old[idx].forces[q] + atom.mass*atom.forces[q]) 

    X.append(abs(mol.atoms[0].origin[0] - mol.atoms[1].origin[0]))
    Y.append(abs(mol.atoms[0].origin[1] - mol.atoms[1].origin[1]))
    Z.append(abs(mol.atoms[0].origin[2] - mol.atoms[1].origin[2]))

plt.plot(np.arange(steps)*dt,np.asarray(Z))
plt.show()






