import numpy as np

G = np.load('dgdbz.npy')

N = len(G)
#N = 3

for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                if not (np.isclose(G[j,i,l,k], G[l,k,j,i]) and (np.isclose(G[i,j,k,l], G[k,l,i,j]) and np.isclose(G[i,j,k,l], -G[j,i,l,k]))): 
                    print i+1,j+1,k+1,l+1

'''
ijkl = 0
ij = 0
for i in range(N):
    for j in range(N):
        ij += 1
        kl = 0
        for k in range(N):
            ik = i + k
            for l in range(N):
                kl += 1
                if ij >= kl and ik >= j+l and not (i==j and k==l):
                    ijkl += 1
                #if not np.isclose(G[i,j,k,l],G2[i,j,k,l]):
#                if i==j and k==l:
#                    print i+1,j+1,k+1,l+1,G[i,j,k,l],G2[i,j,k,l]
#                    print i+1,j+1,k+1,l+1

print ijkl
'''
