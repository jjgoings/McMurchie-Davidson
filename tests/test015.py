import numpy as np
from mmd.slater import * 

def test_simple_slater_condon():
    det_dict = {0b111: 1, 0b1101: -1, 0b1110: 1, 0b10101: -1, 0b10110: 1, 0b11100: 1, 0b100101: -1, 0b100110: 1, 0b101100: 1, 0b110100: 1, 0b001011: 1, 0b010011: 1, 0b011010: -1, 0b100011: 1, 0b101010: -1, 0b110010: -1, 0b011001: 1, 0b101001: 1, 0b110001: 1}
    det1 = np.array([[0b111,0b111]])
    Nint = 1
    for key in det_dict.keys():
        det2 = np.array([[key,0b111]])
        exc, degree, phase = get_excitation(det1,det2,Nint)
        assert phase == det_dict[key]

