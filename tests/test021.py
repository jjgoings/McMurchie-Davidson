import numpy as np
from mmd.utils.davidson import davidson 

def test_davidson():

    np.random.seed(0)
    dim = 1000
    A = np.diag(np.arange(dim,dtype=np.float64))
    A[1:3,1:3] = 0
    M = np.random.randn(dim,dim)
    M += M.T
    A += 1e-4*M

    roots = 5
    E, C = davidson(A, roots)

    E_true, C_true = np.linalg.eigh(A)
    E_true, C_true = E_true[:roots], C_true[:,:roots]

    assert np.allclose(E, E_true)
