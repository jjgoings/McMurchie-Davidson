import numpy as np
from mmd.molecule import Molecule
from mmd.postscf import PostSCF

def test_tdhf():

    water = """
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000
    """
    
    # init molecule and build integrals
    mol = Molecule(geometry=water,basis='sto-3g')
    
    # do the SCF
    mol.RHF()

    # now do TDHF
    PostSCF(mol).TDHF() 
    
    # G16 reference UTDHF excitation energies (full; no FC, 50-50, nosym)
    # note I have manually expanded triplets in G16 to the full degeneracy
    # because at this moment we don't handle spin conserved excitations

    '''
    # td(nstates=100,50-50,full) nosymm uhf/sto-3g int=acc2e=14
    
    test
    
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000

    '''

    gau_ref = [7.7597,7.7597,7.7597,8.1564,8.1564,8.1564,9.5955,9.5955,9.5955,\
               9.6540] 

    assert np.allclose(np.asarray(gau_ref),mol.tdhf_omega[:10])

    ref_omega = mol.tdhf_omega

    # check consistency in solution algorithms for TDHF
    PostSCF(mol).TDHF(alg='reduced')
    assert np.allclose(mol.tdhf_omega,ref_omega)
    PostSCF(mol).TDHF(alg='full')
    assert np.allclose(mol.tdhf_omega,ref_omega)
