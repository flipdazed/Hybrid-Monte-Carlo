#!/usr/bin/env python
import numpy as np

from results.common import acorr as routine
from hmc.potentials import Klein_Gordon as KG

from theory.autocorrelations import M2_Exp as Theory
from theory.operators import magnetisation_sq
from theory.acceptance import acceptance
try:
    from theory.clibs.autocorrelations.exponential import hmc
    theory = True
except:
    print 'failed to import hmc c++ thory'
    theory = False

m         = 1.
n_steps   = 20
step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps

th = Theory(tau=step_size*n_steps, m=m)

file_name = __file__
pot = KG(m=m)

n, dim  = 1000, 1
x0 = np.random.random((n,)*dim)
spacing = 0.2

n_samples, n_burn_in = 10000, 50
c_len   = 100

mixing_angles = [.5*np.pi]
angle_labels = [r'\frac{\pi}{2}']

separations = range(c_len)
opFn = magnetisation_sq
op_name = r'$\hat{X} = \phi_0^2 :\phi_0 = \mathcal{F}^{-1}\phi_0 = '\
    + r' N^{-1}\sum_{x\in\mathbb{Z}^d_\text{N}\phi_0$'
pacc_theory = acceptance(dtau=step_size, tau=tau, n=x0.size, m=pot.m)

if theory: 
    acFunc = lambda t, pa, theta: hmc(pa, tau*m, 1./tau, t)
    op_theory   = hmc(pacc_theory, tau*m, 1./tau, 0)
else:
    acFunc = op_theory = None

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
        save = False)
