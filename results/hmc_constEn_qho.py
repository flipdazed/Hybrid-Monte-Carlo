#!/usr/bin/env python
import numpy as np

from common import hmc_constEn
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO()

n, dim = 100, 1
x0 = np.random.random((n,)*dim)
x0 = np.zeros((n,)*dim)

if '__main__' == __name__:
    hmc_constEn.main(x0, pot, file_name, save = False,
    n_samples=10, n_burn_in=0,
    step_size=0.1, n_steps=20,
#    mixing_angle=0,
    accept_all = False,
    spacing=1.)
