import os
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from plotter import Pretty_Plotter, PLOT_LOC

from common.hmc_model import Model
from hmc.potentials import Klein_Gordon as KG

import dynamics_sho_rev

#
if __name__ == '__main__':
    
    n_steps = 500
    
    dim = 1 ; n=100
    x0 = np.random.random((n,)*dim)
    
    pot = KG()
    
    # set up the model
    model = Model(x0,
        pot       = pot,
        n_steps   = n_steps, 
        step_size = 0.01,
        spacing   = 1.
        )
    model.dynamics.save_path = True
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p
    x0 = model.sampler.x
    
    print 'Running Model: {}'.format(__file__)
    norm = dynamics_sho_rev.reverseIntegration(p0, x0, model, n_steps)
    print 'Finished Running Model: {}'.format(__file__)
    
    # average change across all sites
    av_norm = norm.mean(axis=1)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    dynamics_sho_rev.plot(av_norm, 
        save = save_name
        # save = False
        )
