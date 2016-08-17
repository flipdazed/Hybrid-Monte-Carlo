#!/usr/bin/env python
import numpy as np

from common import metropolis_accRej
from hmc.potentials import Klein_Gordon as KG

m = 0.1
spacing =0.25
n_steps = 30
file_name = __file__
pot = KG(m=m)

n, dim = 50, 1
x0 = np.random.random((n,)*dim)

n_burn_in, n_samples = 10, 100

ac={'store_acceptance':True, 'accept_all':False}

if __name__ == '__main__':
    metropolis_accRej.main(x0, pot, file_name, n_samples, n_burn_in,
        save = True, step_size=1./np.sqrt(3)/m/n_steps, n_steps=n_steps, spacing=spacing, accept_kwargs=ac)
