#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis

from common.hmc_model import Model
from hmc.potentials import Multivariate_Gaussian as MVG

def plot(burn_in, samples, bg_xyz):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp.params['figure.subplot.top'] = 0.85
    pp._updateRC()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel(r'$\mathrm{x_1}$')
    ax.set_ylabel(r'$\mathrm{x_2}$')
    
    fig.suptitle(r'Sampling Multivariate Gaussian with HMC',
        fontsize=pp.ttfont)
    ax.set_title(r'Showing the burn-in \& first 100 HMC moves for:\ $\mu=\begin{pmatrix}0 & 0\end{pmatrix}$, $\Sigma = \begin{pmatrix} 1.0 & 0.8\\ 0.8 & 1.0 \end{pmatrix}$',
        fontsize=(pp.tfont-4))
    
    plt.grid(True)
    
    x, y, z = bg_xyz
    # z = np.ma.array(z, mask=z < .01) # optionally mask to white below certain value
    cnt = ax.contourf(x, y, z, 100, cmap=viridis,
        alpha=.3, antialiased=True)
    
    
    l0 = ax.plot(burn_in[0,0], burn_in[0,1], 
        marker='o', markerfacecolor='green'
        )
    
    l1 = ax.plot(burn_in[:,0], burn_in[:,1], 
        color='grey', linestyle='--'
        # marker='o', markerfacecolor='red'
        )
    
    l2 = ax.plot(samples[:,0], samples[:,1],
        color='blue',
        marker='o', markerfacecolor='red'
        )
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def getPotential(potFn, n_points=100):
    """Returns x,y,z of the analytic potential
    
    Required Inputs
        potFn   :: function :: the potential energy function
    
    Optional Inputs
        n_points :: int :: defines the resolution
    
    Expected
        potFn takes a 1x2 column matrix in and returns a point
    """
    
    n_points = 100    # n**2 is the number of points
    x = np.linspace(-5., 5., n_points, endpoint=True)
    x,y = np.meshgrid(x, x)
    
    z = np.exp( -np.asarray(
            [ model.pot.uE(np.matrix([[i], [j]])) \
              for i,j in zip(np.ravel(x), np.ravel(y))]
            ))
    z = np.asarray(z).reshape(n_points, n_points)
    return x,y,z
#
if __name__ == '__main__':
    
    n_burn_in = 15
    n_samples = 100
    
    pot = MVG()
    
    x0 = np.asarray([[-4.],[4.]])
    model = Model(x0, pot=pot)
    
    # adjust for nice plotting
    print "Running Model: {}".format(__file__)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    print "Finished Running Model: {}".format(__file__)
    
    # change shape from (n, 2) -> (2, n)
    samples = model.samples
    burn_in = model.burn_in
    
    xyz = getPotential(pot.uE)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    plot(burn_in=burn_in, samples=samples, bg_xyz=xyz,
        save = save_name
        # save = False
        )
