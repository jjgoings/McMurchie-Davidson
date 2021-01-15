"""
References: An efficient implementation of Slater-Condon rules
Anthony Scemama, Emmanuel Giner
https://arxiv.org/abs/1311.6244

Determinants are represented as [Nintp array, where Nint = floor(nMOs/64) + 1 
Here the determinants are spin-scattered so (alpha, beta) on same string
"""

import numpy as np

def trailz(i):
    ''' Returns the number of trailing zero bits for a given integer i '''
    count = 0
    while ((i & 1) == 0): # while first bitindex is 0
        i = i >> 1        # remove trailing bit index
        count += 1        # increment count and repeat
    return count

def n_excitations(det1,det2,Nint):

    exc = bin(det1[0] ^ det2[0]).count('1') 
    for l in range(1,Nint):
        exc += bin(det1[l] ^ det2[l]).count('1') 

    exc = exc >> 1 # right bitshift by 1 == divide by 2

    return exc

def get_excitation(det1,det2,Nint):
    exc = np.zeros((3,2),dtype=np.int)
    degree = n_excitations(det1,det2,Nint)
    if degree > 2:
        phase = 0
        degree = -1
    elif degree == 2:
        exc, phase = get_double_excitation(det1,det2,Nint)
    elif degree == 1:
        exc, phase = get_single_excitation(det1,det2,Nint)
    elif degree == 0:
        phase = 1
        pass 

    return exc, degree, phase


def get_single_excitation(det1,det2,Nint):
    exc = np.zeros((3,2),dtype=np.int)
    phase_dbl = [1,-1]

    ishift = -63
    for l in range(Nint):
        ishift += 64
        if det1[l] == det2[l]:
            continue
        tmp = det1[l] ^ det2[l]
        particle = tmp & det2[l]
        hole = tmp & det1[l]
        if particle != 0:
            tz = trailz(particle) 
            exc[0,1] = 1
            exc[1,1] = tz + ishift - 1 # index from 0
        if hole != 0:
            tz = trailz(hole)
            exc[0,0] = 1
            exc[1,0] = tz + ishift - 1 # index from 0

        if (exc[0,0] & exc[0,1]) == 1:
            low = min(exc[1,0],exc[1,1])
            high = max(exc[1,0],exc[1,1])
            j = ((low) >> 6)
            n = ((low) & 63)
            k = ((high) >> 6) 
            m = ((high) & 63)
            if j == k:
                nperm = bin(det1[j] & \
                (~(1 << n+1)+1 & (1 << m)-1)).count('1')
            else:
                nperm = bin((det1[k]) & ((1 << m) - 1)).count('1') +\
                        bin((det1[j]) & (~(1 << n+1) + 1)).count('1')

                for i in range(j+1,k):
                    nperm += bin(det1[i]).count('1')              

    phase = phase_dbl[nperm & 1]
    return exc, phase

def get_double_excitation(det1,det2,Nint):
    exc = np.zeros((3,2),dtype=np.int)
    phase_dbl = [1,-1]
    nexc = 0
    nperm = 0
    idx_particle = 0
    idx_hole = 0
    ishift = -63
    for l in range(Nint):
        ishift += 64
        if det1[l] == det2[l]:
            continue
        tmp = det1[l] ^ det2[l]
        particle = tmp & det2[l]
        hole     = tmp & det1[l]
        while particle != 0:
            tz = trailz(particle) 
            nexc += 1
            idx_particle += 1
            exc[0,1] += 1
            exc[idx_particle,1] = tz + ishift - 1 # index from 0
            particle = particle & (particle - 1)
        while hole != 0:
            tz = trailz(hole) 
            nexc += 1
            idx_hole += 1
            exc[0,0] += 1
            exc[idx_hole,0] = tz + ishift - 1 # index from 0
            hole = hole & (hole - 1)
        if nexc == 4:
            break

    for i in range(1,exc[0,0] + 1): 
        assert (i == 1) or (i == 2)
        low = min(exc[i,0],exc[i,1])
        high = max(exc[i,0],exc[i,1])
        j = ((low) >> 6)
        n = ((low) & 63)
        k = ((high) >> 6) 
        m = ((high) & 63)
        if j == k:
            nperm += bin(det1[j] & \
                (~(1 << n+1)+1 & (1 << m)-1)).count('1')
        else:
            nperm += bin((det1[k]) & ((1 << m) - 1)).count('1') +\
                    bin((det1[j]) & (~(1 << n+1) + 1)).count('1')
 
            for l in range(j+1,k):
                nperm += bin(det1[l]).count('1')              

    if exc[0,0] == 2:
        a = min(exc[1,0], exc[1,1])
        b = max(exc[1,0], exc[1,1])
        c = min(exc[2,0], exc[2,1])
        d = max(exc[2,0], exc[2,1])
        if ((c > a) and (c < b) and (d > b)):
            nperm += 1
    
    phase = phase_dbl[nperm & 1]
    return exc, phase

if __name__ == '__main__':
    ''' exc index tests '''
    det1 = np.array([0b111])
    det2 = np.array([0b101010])
    Nint = 1
    exc, degree, phase = get_excitation(det1,det2,Nint)
    # first excitation in double exc is 0->3
    assert exc[1,0] == 0
    assert exc[1,1] == 3
    # second excitation in double exc is 2->5
    assert exc[2,0] == 2
    assert exc[2,1] == 5

    det1 = np.array([0b111])
    det2 = np.array([0b1000101])
    Nint = 1
    exc, degree, phase = get_excitation(det1,det2,Nint)
    # single excitation is 1->6
    assert exc[1,0] == 1
    assert exc[1,1] == 6



