"""
References: An efficient implementation of Slater-Condon rules
Anthony Scemama, Emmanuel Giner
https://arxiv.org/abs/1311.6244

Determinants are represented as (Nint,2) array, where Nint = floor(nMOs/64) + 1 
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

    exc = bin(det1[0,0] ^ det2[0,0]).count('1') +\
          bin(det1[0,1] ^ det2[0,1]).count('1')
  
    for l in range(1,Nint):
        exc = bin(det1[l,0] ^ det2[l,0]).count('1') +\
              bin(det1[l,1] ^ det2[l,1]).count('1')

    exc = exc >> 1 # right bitshift by 1 == divide by 2

    return exc

def get_excitation(det1,det2,Nint):
    exc = np.zeros((3,2,2),dtype=np.int)
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
    exc = np.zeros((3,2,2),dtype=np.int)
    phase_dbl = [1,-1]

    for ispin in range(2):
        ishift = -63
        for l in range(Nint):
            ishift += 64
            if det1[l,ispin] == det2[l,ispin]:
                continue
            tmp = det1[l,ispin] ^ det2[l,ispin]
            particle = tmp & det2[l,ispin]
            hole = tmp & det1[l,ispin]
            if particle != 0:
                tz = trailz(particle) 
                exc[0,1,ispin] = 1
                exc[1,1,ispin] = tz + ishift
            if hole != 0:
                tz = trailz(hole)
                exc[0,0,ispin] = 1
                exc[1,0,ispin] = tz + ishift

            if (exc[0,0,ispin] & exc[0,1,ispin]) == 1:
                low = min(exc[1,0,ispin],exc[1,1,ispin])
                high = max(exc[1,0,ispin],exc[1,1,ispin])
                j = ((low-1) >> 6)
                n = ((low-1) & 63)
                k = ((high-1) >> 6) 
                m = ((high-1) & 63)
                if j == k:
                    nperm = bin(det1[j,ispin] & \
                    (~(1 << n+1)+1 & (1 << m)-1)).count('1')
                else:
                    nperm = bin((det1[k,ispin]) & ((1 << m) - 1)).count('1') +\
                            bin((det1[j,ispin]) & (~(1 << n+1) + 1)).count('1')

                    for i in range(j+1,k):
                        nperm += bin(det1[i,ispin]).count('1')              

    phase = phase_dbl[nperm & 1]
    return exc, phase

def get_double_excitation(det1,det2,Nint):
    exc = np.zeros((3,2,2),dtype=np.int)
    phase_dbl = [1,-1]
    nexc = 0
    nperm = 0
    for ispin in range(2):
        idx_particle = 0
        idx_hole = 0
        ishift = -63
        for l in range(Nint):
            ishift += 64
            if det1[l,ispin] == det2[l,ispin]:
                continue
            tmp = det1[l,ispin] ^ det2[l,ispin]
            particle = tmp & det2[l,ispin]
            hole     = tmp & det1[l,ispin]
            while particle != 0:
                tz = trailz(particle) 
                nexc += 1
                idx_particle += 1
                exc[0,1,ispin] += 1
                exc[idx_particle,1,ispin] = tz+ishift
                particle = particle & (particle - 1)
            while hole != 0:
                tz = trailz(hole) 
                nexc += 1
                idx_hole += 1
                exc[0,0,ispin] += 1
                exc[idx_hole,0,ispin] = tz+ishift
                hole = hole & (hole - 1)
            if nexc == 4:
                break

        for i in range(1,exc[0,0,ispin] + 1): 
            assert (i == 1) or (i == 2)
            low = min(exc[i,0,ispin],exc[i,1,ispin])
            high = max(exc[i,0,ispin],exc[i,1,ispin])
            j = ((low-1) >> 6)
            n = ((low-1) & 63)
            k = ((high-1) >> 6) 
            m = ((high-1) & 63)
            if j == k:
                nperm += bin(det1[j,ispin] & \
                    (~(1 << n+1)+1 & (1 << m)-1)).count('1')
            else:
                nperm += bin((det1[k,ispin]) & ((1 << m) - 1)).count('1') +\
                        bin((det1[j,ispin]) & (~(1 << n+1) + 1)).count('1')
 
                for l in range(j+1,k):
                    nperm += bin(det1[l,ispin]).count('1')              

        if exc[0,0,ispin] == 2:
            a = min(exc[1,0,ispin], exc[1,1,ispin])
            b = max(exc[1,0,ispin], exc[1,1,ispin])
            c = min(exc[2,0,ispin], exc[2,1,ispin])
            d = max(exc[2,0,ispin], exc[2,1,ispin])
            if ((c > a) and (c < b) and (d > b)):
                nperm += 1
            break
        
    phase = phase_dbl[nperm & 1]
    return exc, phase

