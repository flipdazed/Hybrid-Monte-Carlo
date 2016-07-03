import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from correlations import corr
from hmc_model import Model
from utils import saveOrDisplay

from plotter import Pretty_Plotter, PLOT_LOC

#
def main(x0, pot, file_name, n_samples, n_burn_in, spacing = 1., save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        
    Optional Inputs
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    
    model = Model(x0, pot=pot, spacing=spacing)
    c = corr.Corellations_1d(model, 'run', 'samples')
    
    print 'Running Model: {}'.format(file_name)
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
    av_x_sq = c.twoPoint(separation=0)
    print 'Finished Running Model: {}'.format(file_name)
    
    print '<x(0)x(0)> = {}'.format(av_x_sq)