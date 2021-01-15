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
                exc[1,1,ispin] = tz + ishift - 1 # make 0-indexed
            if hole != 0:
                tz = trailz(hole)
                exc[0,0,ispin] = 1
                exc[1,0,ispin] = tz + ishift - 1 # make 0-indexed

            if (exc[0,0,ispin] & exc[0,1,ispin]) == 1:
                low = min(exc[1,0,ispin],exc[1,1,ispin])
                high = max(exc[1,0,ispin],exc[1,1,ispin])
                j = ((low) >> 6)
                n = ((low) & 63)
                k = ((high) >> 6) 
                m = ((high) & 63)
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
                exc[idx_particle,1,ispin] = tz + ishift - 1 # make 0-indexed
                particle = particle & (particle - 1)
            while hole != 0:
                tz = trailz(hole) 
                nexc += 1
                idx_hole += 1
                exc[0,0,ispin] += 1
                exc[idx_hole,0,ispin] = tz + ishift - 1 # make 0-indexed
                hole = hole & (hole - 1)
            if nexc == 4:
                break

        for i in range(1,exc[0,0,ispin]+1): 
            #assert (i == 1) or (i == 2)
            low = min(exc[i,0,ispin],exc[i,1,ispin])
            high = max(exc[i,0,ispin],exc[i,1,ispin])
            j = ((low) >> 6)
            n = ((low) & 63)
            k = ((high) >> 6) 
            m = ((high) & 63)
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

if __name__ == '__main__':
    det1 = np.array([[0b111,0b111]])
    det2 = np.array([[0b101010,0b111]])
    Nint = 1
    exc, degree, phase = get_excitation(det1,det2,Nint)
    # first excitationi in double exc is alpha 0->3
    assert exc[1,0,0] == 0
    assert exc[1,1,0] == 3
    # second excitation in double exc is alpha 2->5
    assert exc[2,0,0] == 2
    assert exc[2,1,0] == 5

    det1 = np.array([[0b111,0b111]])
    det2 = np.array([[0b111,0b1000101]])
    Nint = 1
    exc, degree, phase = get_excitation(det1,det2,Nint)
    # single beta excitation is beta 1->6
    assert exc[1,0,1] == 1
    assert exc[1,1,1] == 6

    det1 = np.array([[0b1011,0b111]])
    det2 = np.array([[0b111,0b10101]])
    Nint = 1
    exc, degree, phase = get_excitation(det1,det2,Nint)
    # alpha single excitation is alpha 3->2
    assert exc[1,0,0] == 3
    assert exc[1,1,0] == 2
    # beta single excitation is beta 1->4
    assert exc[1,0,1] == 1
    assert exc[1,1,1] == 4

    det_dict = {0b111: 1, 0b1101: -1, 0b1110: 1, 0b10101: -1, 0b10110: 1, 0b11100: 1, 0b100101: -1, 0b100110: 1, 0b101100: 1, 0b110100: 1, 0b001011: 1, 0b010011: 1, 0b011010: -1, 0b100011: 1, 0b101010: -1, 0b110010: -1, 0b011001: 1, 0b101001: 1, 0b110001: 1}
    det1 = np.array([[0b111,0b111]])
    Nint = 1
    for key in det_dict.keys():
        det2 = np.array([[key,0b111]])
        exc, degree, phase = get_excitation(det1,det2,Nint)
#        print(det1,det2,exc,phase)
        assert phase == det_dict[key]


