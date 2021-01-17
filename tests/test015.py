import numpy as np
from mmd.slater import * 

def test_simple_slater_condon():
    ''' phase tests for CISD on (3,6) space'''
    det_dict = {0b111: 1, 0b1101: -1, 0b1110: 1, 0b10101: -1, 0b10110: 1, 0b11100: 1, 0b100101: -1, 0b100110: 1, 0b101100: 1, 0b110100: 1, 0b001011: 1, 0b010011: 1, 0b011010: -1, 0b100011: 1, 0b101010: -1, 0b110010: -1, 0b011001: 1, 0b101001: 1, 0b110001: 1}
    det1 = np.array([0b111])
    Nint = 1
    for key in det_dict.keys():
        det2 = np.array([key])
        exc, degree, phase = get_excitation(det1,det2,Nint)
        assert phase == det_dict[key]


def test_exc_index():
    ''' exc index tests '''
    det1 = np.array([0b111])
    det2 = np.array([0b101010])
    Nint = 1
    exc, degree, phase = get_excitation(det1,det2,Nint)
    print(exc)
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
    print(exc)
    assert exc[1,0] == 1
    assert exc[1,1] == 6

def test_common_index():
    det1 = np.array([0b111,0b101,0b11000])
    det2 = np.array([0b1110,0b1101,0b11000])
    common = common_index(det1,det2,3)
    assert set(common) == set([1,2,64,66,131,132])
