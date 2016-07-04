
import numpy as np

from common import corr1d_x2_0
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO(m0=1., mu=1.)

n, dim = 51, 1
x0 = np.random.random((n,)*dim)
spacing = 1.


if '__main__' == __name__:
    corr1d_x2_0.main(x0, pot, file_name,
        n_samples = 1000, n_burn_in = 50, spacing = spacing,
        save = False)