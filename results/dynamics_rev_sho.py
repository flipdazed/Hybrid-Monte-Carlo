#!/usr/bin/env python
import numpy as np

from common import dynamics_rev
from hmc.potentials import Simple_Harmonic_Oscillator as SHO

file_name = __file__
pot = SHO()


x0 = np.asarray([[1.]])

n_steps = int(5e6)

if '__main__' == __name__:
    dynamics_rev.main(x0, pot, file_name, save = True, n_steps = n_steps, progress_bar = True)