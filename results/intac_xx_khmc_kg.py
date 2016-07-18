import numpy as np

from common import multiangle
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint

file_name = __file__
pot = KG()

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing    = 1.
step_size  = .1
rand_steps = True
points     = 100

n_samples, n_burn_in = 20000, 20
angle_fracs = np.linspace(0, .1, points, True)

opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O}_{pq} = \phi_0^2$'

if '__main__' == __name__:
    multiangle.main(x0, pot, file_name, 
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        rand_steps = rand_steps, step_size = step_size, n_steps = 1,
        opFn = opFn, op_name=op_name,
        angle_fracs = angle_fracs,
        save = True)