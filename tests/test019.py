import numpy as np
from mmd.molecule import Molecule
from mmd.postscf import PostSCF

def test_cis():

    water = """
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000
    """
    
    # init molecule and build integrals
    mol = Molecule(geometry=water,basis='3-21G')
    
    # do the SCF
    mol.RHF()

    # now do FCI
    PostSCF(mol).CIS() 
    
    # G16 reference UCIS excitation energies (full; no FC, 50-50, nosym)
    # note I have manually expanded triplets in G16 to the full degeneracy
    # because at this moment we don't handle spin conserved excitations

    '''
    #p cis(nstate=40,full) uhf/3-21G int=acc2e=14 nosymm

    water
    
    0 1
    O    0.000000      -0.075791844    0.000000
    H    0.866811829    0.601435779    0.000000
    H   -0.866811829    0.601435779    0.000000

    '''

    gau_ref = [6.5822,6.5822,6.5822,7.7597,7.8156,7.8156,7.8156,8.4377,8.4377, \
               8.4377, 9.0903,9.0903,9.0903,9.3334,10.5493]

    assert np.allclose(np.asarray(gau_ref),mol.cis_omega[:15])

