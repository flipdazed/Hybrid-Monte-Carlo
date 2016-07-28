#!/usr/bin/env python
import numpy as np

from common import acorr as routine
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint
from theory.autocorrelations import M2_Exp as Theory

file_name = __file__
pot = KG(m=1)

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 1000, 25
step_size =  .1
n_steps   =  100
c_len     = 200

mixing_angles = [.5*np.pi]
angle_labels = [r'\frac{\pi}{2}']

separations = range(c_len)
opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O} = \sum_{pq} \Omega \phi_p\phi_q :\Omega = \delta_{p0}\delta_{q0}$'

# this intiial declaration doesn't really matter
# as eval() re-evaluates every time anyway
th = Theory(tau=(step_size*n_steps), m=1)

if '__main__' == __name__:
    routine.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        mixing_angles=mixing_angles, angle_labels = angle_labels,
        separations = separations, opFn = opFn, op_name = op_name,
        rand_steps= True, step_size = step_size, n_steps = n_steps,
        Theory_Cls = th,
        save = False)
