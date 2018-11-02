from __future__ import division
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from mmd.integrals.twoe import ERI
#include "util.pxi"


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def formPT(np.ndarray[complex, ndim=2] P, np.ndarray[complex, ndim=2] P_old, list bfs, long nbasis, dict screen, double tol):
    """Routine to build the AO basis Fock matrix"""
    cdef:
        long N = nbasis
        long i,j,k,l,ij,il
        double s12_deg, s34_deg, s12_34_deg, s1234_deg
        double eri, bound, dmax
        #complex [:,:] G
    # perturbation tensor and density difference
    G = np.zeros((N,N),dtype='complex')
    
    cdef np.ndarray[complex, ndim=2] dP = P - P_old
    # Comments from Liblong hartreefock++
    #  1) each shell set of integrals contributes up to 6 shell sets of
    #  the Fock matrix:
    #     F(a,b) += (ab|cd) * D(c,d)
    #     F(c,d) += (ab|cd) * D(a,b)
    #     F(b,d) -= 1/4 * (ab|cd) * D(a,c)
    #     F(b,c) -= 1/4 * (ab|cd) * D(a,d)
    #     F(a,c) -= 1/4 * (ab|cd) * D(b,d)
    #     F(a,d) -= 1/4 * (ab|cd) * D(b,c)
    #  2) each permutationally-unique integral (shell set) must be
    #  scaled by its degeneracy,
    #     i.e. the number of the integrals/sets equivalent to it
    #  3) the end result must be symmetrized
    for i in range(N):
        for j in range(i+1):
            ij = (i*(i+1)//2 + j)
            for k in range(N):
                for l in range(k+1):
                    kl = (k*(k+1)//2 + l)
                    if ij >= kl:
                        # use cauchy-schwarz to screen
                        bound = (sqrt(screen[ij])
                                *sqrt(screen[kl]))
                        # screen based on contr. with density diff
                        dmax = np.max(np.abs([4*dP[i,j],
                                       4*dP[k,l],
                                         dP[i,k],
                                         dP[i,l],
                                         dP[j,k],
                                         dP[j,l]]))
                        bound *= dmax
                        if bound < tol:
                            continue
                        else:
                            # work out degeneracy scaling
                            s12_deg = 1.0 if (i == j) else 2.0
                            s34_deg = 1.0 if (k == l) else 2.0
                            if i == k:
                                if j == l:
                                    s12_34_deg = 1.0
                                else:
                                    s12_34_deg = 2.0
                            else:
                                s12_34_deg = 2.0

                            s1234_deg = s12_deg * s34_deg * s12_34_deg

                            #FIXME: should use integrals from screen
                            # rather than re-compute the values
                            eri = s1234_deg*ERI(bfs[i],bfs[j],
                                                bfs[k],bfs[l])

                            # See Almlof, Faegri, Korsell, 1981
                            # Coulomb, Eq (4a,4b) of Korsell, 1981 
                            G[i,j] += dP[k,l]*eri
                            G[k,l] += dP[i,j]*eri
                            # Exchange, Eq (5) of Korsell, 1981 
                            G[i,k] += -0.25*dP[j,l]*eri
                            G[j,l] += -0.25*dP[i,k]*eri
                            G[i,l] += -0.25*dP[j,k]*eri
                            G[k,j] += -0.25*dP[i,l]*eri

    return np.asarray(G)
