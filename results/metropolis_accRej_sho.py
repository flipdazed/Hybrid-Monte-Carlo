#!/usr/bin/env python
import numpy as np

from common import metropolis_accRej
from hmc.potentials import Simple_Harmonic_Oscillator as SHO

file_name = __file__
pot = SHO()



x0 = np.asarray([[1.]])

n_burn_in, n_samples = 15, 100

if __name__ == '__main__':
    metropolis_accRej.main(x0, pot, file_name, n_samples, n_burn_in,
        save = True)