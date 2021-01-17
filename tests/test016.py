import numpy as np
from mmd.molecule import Molecule
from mmd.postscf import PostSCF

def test_mp2_spinorbital():
    helium_dimer = """
    0 1
    He   0.0 0.0 0.0
    He   0.0 0.0 0.75
    """
    
    # init molecule and build integrals
    mol = Molecule(geometry=helium_dimer,basis='cc-pvDZ')
    
    # do the SCF
    mol.RHF()
    
    # do MP2
    PostSCF(mol).MP2()
    emp2_spatial = mol.emp2.real
    PostSCF(mol).MP2(spin_orbital=True)
    emp2_spin = mol.emp2.real
   
    # consistency check 
    assert emp2_spatial == emp2_spin
    
    # G16 reference SCF energy
    assert np.allclose(-5.29648041091,mol.energy.real)
    
    # G16 reference MP2 energy
    assert np.allclose(-5.3545864180140,emp2_spatial)
