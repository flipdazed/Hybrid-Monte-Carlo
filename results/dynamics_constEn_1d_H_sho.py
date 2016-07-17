#!/usr/bin/env python
import numpy as np

from common import dynamics_constEn_1d
from hmc.potentials import Simple_Harmonic_Oscillator as SHO

file_name = __file__
pot = SHO()



x0 = np.asarray([[1.]])



if '__main__' == __name__:
    dynamics_constEn_1d.main(x0, pot, file_name, save = False, all_lines = False)