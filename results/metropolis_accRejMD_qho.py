#!/usr/bin/env python
import numpy as np

from common import metropolis_accRej
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO()

# lattice
n, dim      = 100, 1    # number of sites and dimensions

# calculations
shape       = (n,)*dim
x0 = np.random.random(shape)

ac={'store_acceptance':True, 'accept_all':False}
if '__main__' == __name__:
    metropolis_accRej.main(x0, pot, file_name, n_samples=10, n_burn_in=0,
                accept_kwargs=ac, 
                step_size=0.05, 
                n_steps=20, 
                spacing=1., 
                mixing_angle=np.pi/2., 
                accept_all = False,
                plot_mdmc=True,
                 save = True)
