import numpy as np
import scipy

def davidson(A,roots,tol=1e-8):

    mat_dim = A.shape[0]
    sub_dim = 4*roots
    V = np.eye(mat_dim,sub_dim) 

    converged = False
    while not converged:
        # subspace
        S = np.dot(V.T,np.dot(A,V))
 
        # diag subspace
        E,C = scipy.linalg.eigh(S) # note already ordered if using eigh
        E,C = E[:roots], C[:,:roots]
         
        # get current approx. eigenvectors
        X = np.dot(V,C)   

        # form residual vectors 
        R = np.zeros((mat_dim,roots))
        Delta = np.zeros((mat_dim,roots))
        unconverged = []
        for j in range(roots):
            R[:,j] = np.dot((A - E[j] * np.eye(mat_dim)),X[:,j])
            if np.linalg.norm(R[:,j]) > tol:
                unconverged.append(j)

        # check convergence
        if len(unconverged) < 1:
            converged = True

        for j in unconverged:
            preconditioner = np.zeros(mat_dim)
            for i in range(mat_dim):
                if np.abs(E[j] - A[i,i]) < 1e-4:
                    continue # don't let preconditioner blow up -- keep as zero
                else:
                    preconditioner[i] = -1/(A[i,i] - E[j])

            # normalize correction vectors
            Delta[:,j] = preconditioner * R[:,j]
            Delta[:,j] /= np.linalg.norm(Delta[:,j])

            # project corrections onto orthogonal complement
            q = np.dot((np.eye(mat_dim)  - np.dot(V,V.T)),Delta[:,j])
            norm = np.linalg.norm(q)

            if (norm > 1e-4):
                if (sub_dim + 1 > min(500,mat_dim//4)):
                    # subspace collapse 
                    print("Subspace too big: collapsing")
                    print("Eigs at current: ", E)
                    sub_dim = roots # restart uses best guess of eigenvecs
                    V = X
                    V, _ = np.linalg.qr(V)
                    break
                    
                else:
                    V_copy = np.copy(V)
                    sub_dim += 1
                    V = np.eye(mat_dim,sub_dim) 
                    V[:,:(sub_dim-1)] = V_copy
                    # add new vector to end of subspace
                    V[:,-1] = q/norm

    if converged:
        return E, X 

