
import numpy as np

from common import acorr1d_x2
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO(m0=1., mu=1.)

n, dim = 21, 1
x0 = np.random.random((n,)*dim)
spacing = 1.


if '__main__' == __name__:
    acorr1d_x2.main(x0, pot, file_name,
        n_samples = 100, n_burn_in = 20, spacing = spacing,
        c_len = 10,
        step_size = 0.1, n_steps = 20,
        save = True)
