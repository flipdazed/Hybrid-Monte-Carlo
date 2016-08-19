
import numpy as np

from common import corr1d_xx
from hmc.potentials import Klein_Gordon as KG

file_name = __file__
pot = KG(phi_4=1, m=1)

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 0.5


if '__main__' == __name__:
    corr1d_xx.main(x0, pot, file_name,
        n_samples = 100000, n_burn_in = 25, spacing = spacing,
        c_len = 30,
        step_size = 0.1, n_steps = 20,
        save = True, free=False)
