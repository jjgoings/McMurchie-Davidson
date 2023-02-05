import numpy as np
from mmd.molecule import Molecule
from mmd.postscf import PostSCF

def test_cis_bitstring():

    water = """
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000
    """
    
    #test STO-3G oscillator strengths
    mol = Molecule(geometry=water,basis='STO-3G')
    
    # do the SCF
    mol.RHF()

    # now do FCI
    PostSCF(mol).CIS() 
    osc_strength = mol.cis_oscil 

    PostSCF(mol).CIS(bitstring=True)

    assert np.allclose(osc_strength, mol.cis_oscil)

    #test 3-21G oscillator strengths
    mol = Molecule(geometry=water,basis='3-21G')
    # do the SCF
    mol.RHF()

    PostSCF(mol).CIS() 
    osc_strength = mol.cis_oscil 

    PostSCF(mol).CIS(bitstring=True)

    assert np.allclose(osc_strength, mol.cis_oscil)

    #test DZ oscillator strengths
    mol = Molecule(geometry=water,basis='DZ')
    # do the SCF
    mol.RHF()

    PostSCF(mol).CIS() 
    osc_strength = mol.cis_oscil 

    PostSCF(mol).CIS(bitstring=True)

    assert np.allclose(osc_strength, mol.cis_oscil)

    #Test H2 STO_3G
    h2 = """
    0 1
    H    0.000000  0.000000  0.000000
    H    0.740000  0.000000  0.000000
    """
    
    #test STO-3G oscillator strengths
    mol = Molecule(geometry=h2,basis='STO-3G')
    
    # do the SCF
    mol.RHF(DIIS=False)

    # now do FCI
    PostSCF(mol).CIS() 
    osc_strength = mol.cis_oscil 

    PostSCF(mol).CIS(bitstring=True)

    assert np.allclose(osc_strength, mol.cis_oscil)
   
    #test STO-3G oscillator strengths
    mol = Molecule(geometry=h2,basis='3-21G')
    
    # do the SCF
    mol.RHF(DIIS=False)

    # now do FCI
    PostSCF(mol).CIS() 
    osc_strength = mol.cis_oscil 

    PostSCF(mol).CIS(bitstring=True)

    assert np.allclose(osc_strength, mol.cis_oscil)

test_cis_bitstring()
