#!/usr/bin/env python
import numpy as np

from common import acorr as routine
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint
from theory.autocorrelations import M2_Fix as Theory

file_name = __file__
pot = KG()

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 10000, 25
step_size =   .1
n_steps   =  1
c_len     = 1000

mixing_angles = 1/np.asarray([8, 6, 4, 3, 2], dtype='float64')*np.pi
angle_labels = [
    # r'$\theta = \pi/10$',
    r'\pi/8',
    r'\pi/6',
    r'\pi/4',
    r'\pi/3',
    r'\pi/2'
]

separations = range(c_len)
opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O} = \sum_{pq} \Omega \phi_p\phi_q :\Omega = \delta_{p0}\delta_{q0}$'

# this intiial declaration doesn't really matter
# as eval() re-evaluates every time anyway
th = Theory(tau = n_steps*step_size, m=pot.mu)
acFunc = lambda pt: th.eval(t=fx, pa=pt[0], theta=pt[1]) / th.eval(t=0, pa=pt[0], theta=pt[1])

if '__main__' == __name__:
    routine.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        mixing_angles=mixing_angles, angle_labels = angle_labels,
        separations = separations, opFn = opFn, op_name = op_name,
        step_size = step_size, n_steps = n_steps,
        acFunc = acFunc,
        save = False)