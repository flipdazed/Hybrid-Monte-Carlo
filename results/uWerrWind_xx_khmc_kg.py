#!/usr/bin/env python
import numpy as np

from common import uWerrWind
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint

file_name = __file__
pot = KG()

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 20000, 100
step_size =   .1
n_steps   =  1

mixing_angle = 1/np.asarray([10., 8, 6.])*np.pi
angle_labels = [
    r'$\theta = \pi/10$',
    r'$\theta = \pi/8$',
    r'$\theta = \pi/6$',
    # r'$\theta = \pi/5$',
    # r'$\theta = \pi/4$',
    # r'$\theta = \pi/2$'
]

opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\langle \phi^2 \rangle_{L}$'

if '__main__' == __name__:
    uWerrWind.main(x0, pot, file_name, 
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        rand_steps = True, step_size = step_size, n_steps = n_steps, 
        mixing_angle = mixing_angle, angle_labels = angle_labels, 
        opFn = opFn, op_name=op_name,
        save = True)