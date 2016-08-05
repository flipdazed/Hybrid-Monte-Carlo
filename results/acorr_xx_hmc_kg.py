#!/usr/bin/env python
import numpy as np

from results.common import acorr as routine
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint
from theory.autocorrelations import M2_Exp as Theory
from theory.operators import phi2_1df
from theory.acceptance import acceptance

m = 1.0
n_steps   = 1
step_size = 1/((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps

tau = n_steps*step_size
th = Theory(tau=step_size*n_steps, m=m)

file_name = __file__
pot = KG(m=m)

n, dim = 10, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 100000, 25
c_len     = 10000

mixing_angles = [.5*np.pi]
angle_labels = [r'\frac{\pi}{2}']

separations = range(c_len)
opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O} = \sum_{pq} \Omega \phi_p\phi_q :\Omega = \delta_{p0}\delta_{q0}$'
op_theory   = phi2_1df(mu=pot.m, n=x0.size, a=spacing, sep=0)
pacc_theory = acceptance(dtau=step_size, tau=tau, n=x0.size, m=pot.m, t=tau*n_samples)

th = Theory(tau = tau)
acFunc = th.eval

print '> Trajectory Length: tau: {}'.format(tau)
print '> Step Size: {}'.format(step_size)
print '> Theoretical <x^2>: {}'.format(op_theory)
print '> Theoretical <P_acc>: {}'.format(pacc_theory)

if '__main__' == __name__:
    routine.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        mixing_angles=mixing_angles, angle_labels = angle_labels,
        separations = separations, opFn = opFn, op_name = op_name,
        rand_steps= True, step_size = step_size, n_steps = n_steps,
        acFunc = acFunc,
        save = True)
