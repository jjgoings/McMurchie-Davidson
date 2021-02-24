import numpy as np
import sys

def davidson(A,roots,tol=1e-6):

    mat_dim = A.shape[0]
    sub_dim = 4*roots
    V = np.eye(mat_dim,sub_dim) 

    converged = False
    while not converged:
        # subspace
        S = V.T @ A @ V
 
        # diag subspace
        E,C = np.linalg.eigh(S) # note already ordered if using eigh
        E,C = E[:roots], C[:,:roots]
         
        # get current approx. eigenvectors
        X = V @ C    

        # calc residuals
        R = np.zeros((mat_dim,roots))
        Delta = np.zeros((mat_dim,roots))
        for j in range(roots):
            R[:,j] = (A - E[j] * np.eye(mat_dim)) @ X[:,j]
            preconditioner = np.zeros(mat_dim)
            for i in range(mat_dim):
                if np.abs(E[j] - A[i,i]) < 1e-4:
                    continue # don't let preconditioner blow up -- keep as zero
                else:
                    preconditioner[i] = 1/(E[j] - A[i,i])

            # normalize correction vectors
            Delta[:,j] = preconditioner * R[:,j]
            Delta[:,j] /= np.linalg.norm(Delta[:,j])

        # project corrections onto orthogonal complement
        for j in range(roots):
            q = (np.eye(mat_dim)  - V @ V.T) @ Delta[:,j]
            norm = np.linalg.norm(q)

            if (norm > 1e-3):
                if (sub_dim + 1 > mat_dim):
                    sys.exit("sub_dim > mat_dim")

                V_copy = np.copy(V)
                sub_dim += 1
                V = np.eye(mat_dim,sub_dim) 
                V[:,:(sub_dim-1)] = V_copy
                # add new vector to end of subspace
                V[:,-1] = q/norm
       
        # check convergence
        converged = True
        for j in range(roots):
            if np.linalg.norm(R[:,j]) > tol:
                converged = False     

    if converged:
        return E, C

