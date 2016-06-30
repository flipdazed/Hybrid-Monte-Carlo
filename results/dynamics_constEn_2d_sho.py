#!/usr/bin/env python
import numpy as np

from common import dynamics_constEn_2d
from hmc.potentials import Simple_Harmonic_Oscillator as SHO

file_name = __file__
pot = SHO()


x0 = np.asarray([[1.]])

res = 150

if '__main__' == __name__:
    dynamics_constEn_2d.main(x0, pot, file_name, save = True,
        n_steps = res, n_sizes = res, log_scale=False)