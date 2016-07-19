#!/usr/bin/env python
import numpy as np

from common import errs
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint

file_name = __file__
pot = KG()

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 20000, 25
step_size  =  .1
n_steps    = 20

mixing_angle = np.pi/2.
angle_label = r'$\theta = \pi/2$'
opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\langle \phi^2 \rangle_{L}$'

if '__main__' == __name__:
    errs.main(x0, pot, file_name, 
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        rand_steps = True, step_size = step_size, n_steps = n_steps, 
        mixing_angle = mixing_angle, angle_labels = angle_label, 
        opFn = opFn, op_name=op_name,
        save = True)