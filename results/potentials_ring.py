#!/usr/bin/env python
import numpy as np
import os
import matplotlib.pyplot as plt

from hmc.potentials import Ring_Potential
from plotter import Pretty_Plotter, viridis, magma, inferno, plasma, PLOT_LOC

def plot(x, y, z, save):
    """Plots a test image of the Bivariate Gaussian
    
    Required Inputs
        x,y,z   :: arrays        :: must all be same length
        save    :: string / bool :: save location or '' / False
    """
    
    pp = Pretty_Plotter()
    pp.s = 1.0
    pp._teXify() # LaTeX
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    # create contour plot
    c = ax.contourf(x, y, z, 200, cmap=magma)
    
    # axis labels
    ax.set_xlabel(r'$\phi_1$', fontsize=pp.axfont)
    ax.set_ylabel(r'$\phi_2$', fontsize=pp.axfont)
    ax.grid(False) # remove grid
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    print "Running Model: {}".format(__file__)
    
    # n**2 is the number of points
    # defines the resolution of the plot
    n = 200
    
    pot = Ring_Potential()
    
    # create a mesh grid of NxN
    x = np.linspace(-10., 10., n, endpoint=True)
    x,y = np.meshgrid(x,x)
    
    # ravel() flattens the arrays into 1D vectors
    # and then they are passed as (x,y) components to the
    # potential term to form z = f(x,y)
    z = [np.exp(-pot.uE(np.asarray([[i],[j]]))) for i,j in zip(np.ravel(x), np.ravel(y))]
    
    # reshape back into an NxN
    z = np.asarray(z).reshape(n, n)
    print "Finished Running Model: {}".format(__file__)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    plot(x,y,z,
        save = save_name
        # save = False
        )
    
