import numpy as np
from mmd.molecule import Molecule
from mmd.postscf import PostSCF

def test_spin_adapted_fci():

    water = """
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000
    """
    
    # init molecule and build integrals
    mol = Molecule(geometry=water,basis='STO-3G')
    
    # do the SCF
    mol.RHF()

    # now do FCI
    PostSCF(mol).FCI(spin_adapt=False) 
    e_fci = mol.efci

    #do spin-adapted
    PostSCF(mol).FCI(spin_adapt=True)

    assert np.isclose(e_fci, mol.efci)

    h2 = """
    0 1
    H    0.000000   0.000000    0.000000
    H    0.740000   0.000000    0.000000
    """

    mol = Molecule(geometry=h2,basis='STO-3G')
    
    # do the SCF
    mol.RHF(DIIS=False)

    # now do FCI
    PostSCF(mol).FCI(spin_adapt=False) 
    e_fci = mol.efci

    #do spin-adapted
    PostSCF(mol).FCI(spin_adapt=True)

    assert np.isclose(e_fci, mol.efci)


test_spin_adapted_fci()
   
