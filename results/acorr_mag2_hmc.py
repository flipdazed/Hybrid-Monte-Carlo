#!/usr/bin/env python
import numpy as np

from results.common import acorr as routine
from hmc.potentials import Klein_Gordon as KG

from theory.autocorrelations import M2_Exp as Theory
from theory.operators import magnetisation_sq
from theory.acceptance import acceptance
from theory.clibs.autocorrelations.exponential import hmc

m         = 1.0
n_steps   = 20
step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps

th = Theory(tau=step_size*n_steps, m=m)

file_name = __file__
pot = KG(m=m)

n, dim  = 100, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 1000000, 1000
c_len   = 1000

mixing_angles = [.5*np.pi]
angle_labels = [r'\frac{\pi}{2}']

separations = range(c_len)
opFn = magnetisation_sq
op_name = r'$\mathscr{M}^2$'
pacc_theory = acceptance(dtau=step_size, tau=tau, n=x0.size, m=pot.m)

acFunc = lambda t, pa, theta: hmc(pa, tau*m, 1./tau, t)
op_theory   = hmc(pacc_theory, tau*m, 1./tau, 0)

print '> Trajectory Length: tau: {}'.format(tau)
print '> Step Size: {}'.format(step_size)
print '> Theoretical <\phi_0^2>: {}'.format(op_theory)
print '> Theoretical <P_acc>: {}'.format(pacc_theory)

if '__main__' == __name__:
    routine.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        mixing_angles=mixing_angles, angle_labels = angle_labels,
        separations = separations, opFn = opFn, op_name = op_name,
        rand_steps= True, step_size = step_size, n_steps = n_steps,
        acFunc = acFunc,
        save = True)