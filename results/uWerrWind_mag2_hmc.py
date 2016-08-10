#!/usr/bin/env python
import numpy as np

from common import uWerrWind
from hmc.potentials import Klein_Gordon as KG
from theory.operators import magnetisation_sq
from theory.autocorrelations import M2_Exp
from theory.clibs.autocorrelations.exponential import hmc

file_name = __file__
pot = KG()

m         = 1.0
n_steps   = 20
step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps

n, dim = 10, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 1000000, 25

mixing_angle = [np.pi/2.]
angle_label = [r'$\theta = \pi/2$']
opFn = magnetisation_sq
op_name = r'$\hat{O} = \phi_0^2 :\phi_0 = \mathcal{F}^{-1}\tilde{\phi}_0 = '\
    + r' \sum_{x\in\mathbb{Z}^d_\text{L}}\tilde{\phi}_0$'
th = M2_Exp(tau=n_steps*step_size, m=1)
t = th.integrated

acFunc = lambda t, pa, theta: hmc(pa, tau*m, 1./tau, t)/hmc(pa, tau*m, 1./tau, 0)
separations=range(500)

if '__main__' == __name__:
    uWerrWind.main(x0, pot, file_name, 
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        rand_steps = True, step_size = step_size, n_steps = n_steps, 
        mixing_angle = mixing_angle, angle_labels = angle_label, 
        opFn = opFn, op_name=op_name, itauFunc=t, separations=separations,
        acTheory=acFunc,
        save = True)