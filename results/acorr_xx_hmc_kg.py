#!/usr/bin/env python
import numpy as np

from common import acorr as routine
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint
from theory.autocorrelations import M2_Exp as Theory

m = 0.4
step_size = 1/((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/20.
n_steps   = 20

tau = n_steps*step_size
th = Theory(tau=1/(step_size*n_steps), m=m)
print step_size
file_name = __file__
pot = KG(m=m)

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 100000, 25
c_len     = 50000

mixing_angles = [.5*np.pi]
angle_labels = [r'\frac{\pi}{2}']

separations = range(c_len)
opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O} = \sum_{pq} \Omega \phi_p\phi_q :\Omega = \delta_{p0}\delta_{q0}$'

th = Theory(tau = n_steps*step_size)
acFunc = lambda pt: th.eval(t=fx, pa=pt[0], theta=pt[1]) / th.eval(t=0, pa=pt[0], theta=pt[1])

if '__main__' == __name__:
    routine.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        mixing_angles=mixing_angles, angle_labels = angle_labels,
        separations = separations, opFn = opFn, op_name = op_name,
        rand_steps= True, step_size = step_size, n_steps = n_steps,
        acFunc = acFunc,
        save = False)
