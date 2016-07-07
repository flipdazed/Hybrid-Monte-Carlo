#!/usr/bin/env python
import numpy as np

from common import metropolis_accRej
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO()

n, dim = 100, 1
x0 = np.random.random((n,)*dim)

n_burn_in, n_samples = 10, 50

if __name__ == '__main__':
    metropolis_accRej.main(x0, pot, file_name, n_samples, n_burn_in,
        save = False, step_size=0.1, n_steps=37, spacing=1)
