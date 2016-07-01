#!/usr/bin/env python
import numpy as np

from common import dynamics_rev
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO()

n, dim = 100, 1
x0 = np.random.random((n,)*dim)



if '__main__' == __name__:
    dynamics_rev.main(x0, pot, file_name, save = True)