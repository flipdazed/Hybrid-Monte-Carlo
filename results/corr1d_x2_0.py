
import numpy as np

from common import corr1d_x2_0
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO(m0=1., mu=1.)

n, dim = 100, 1
x0 = np.random.random((n,)*dim)
spacing = .01


if '__main__' == __name__:
    corr1d_x2_0.main(x0, pot, file_name,
        n_samples = 50, n_burn_in = 15, spacing = spacing,
        save = False)
    