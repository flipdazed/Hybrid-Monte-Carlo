import os
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from plotter import Pretty_Plotter, PLOT_LOC

from common.hmc.lattice import Model
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

import dynamics_sho_rev

#
if __name__ == '__main__':
    
    n_steps = 500
    
    pot = QHO()
    # set up the model
    model = Model(
        pot       = pot,
        n_steps   = n_steps, 
        step_size = 0.01,
        spacing   = 1.
        )
    model.dynamics.save_path = True
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p
    x0 = model.x0
    
    print 'Running Model'
    norm = dynamics_sho_rev.reverseIntegration(p0, x0, model, n_steps)
    print 'Finished Running Model'
    
    # average change across all sites
    av_norm = norm.mean(axis=1)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    dynamics_sho_rev.plot(av_norm, 
        # save=False
        save=save_name
        )