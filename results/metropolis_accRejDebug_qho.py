#!/usr/bin/env python
import numpy as np

from common import metropolis_accRej
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO()

# GHMC mixing angle
theta       = np.pi/2

# MDMC
plot_mdmc   = True      # expect to be plotting MDMC
step_size   = 0.1       # step-size
n_steps     = 20        # steps / trajectory

# lattice
n, dim      = 100, 1    # number of sites and dimensions
spacing     = 1         # lattice spacing
cold        = False     # True sets a cold start from 0

# HMC
n_burn_in   = 0         # best to keep 0 to show all paths
n_samples   = 10        # too high clutters plot 10-20 good
accept_all  = False     # disable Metropolis Step by setting to True

# calculations
shape       = (n,)*dim
if cold:
    x0 = np.zeros(shape)
else:
    x0 = np.random.random(shape)

if '__main__' == __name__:
    metropolis_accRej.main(x0, pot, file_name, spacing=spacing,
        n_samples=n_samples, n_burn_in=n_burn_in,  mixing_angle=theta, accept_all = accept_all,
        step_size=step_size, n_steps=n_steps, plot_mdmc = plot_mdmc,
        save = False)
