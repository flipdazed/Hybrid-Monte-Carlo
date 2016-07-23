#!/usr/bin/env python
import numpy as np

from common import intac
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint

file_name = __file__
pot = KG()

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing    = 1.
step_size  = .1
rand_steps = True
points     = 50
n_steps    = 10

n_samples, n_burn_in = 100000, 20
h_res = np.linspace(-0.15, 0.15, 100, True)
l_res = np.linspace(0.151, 0.85, points, True)
angle_fracs = np.concatenate([h_res,l_res,h_res+1,l_res+1,h_res+2])

opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O}_{pq} = \phi_0^2$'

if '__main__' == __name__:
    intac.main(x0, pot, file_name, 
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        rand_steps = rand_steps, step_size = step_size, n_steps = n_steps,
        opFn = opFn, op_name=op_name,
        angle_fracs = angle_fracs,
        save = False)
