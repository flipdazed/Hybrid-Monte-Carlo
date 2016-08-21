#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis

from models import Basic_HMC as Model
from hmc.potentials import Klein_Gordon
from hmc.lattice import Periodic_Lattice

def plot(burn_in, samples, bg_xyz, save):
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
    
    fig.suptitle(r'Sampling a ring potential with HMC',
        fontsize=pp.ttfont)
    ax.set_title(r'Showing the burn-in \& first 100 HMC moves',
        fontsize=(pp.tfont-4))
    
    plt.grid(True)
    
    x, y, z = bg_xyz
    
    small = 1e-10
    mask = z<small
    z = np.ma.MaskedArray(z, mask, fill_value=np.nan)
    x = np.ma.MaskedArray(x, mask, fill_value=np.nan)
    y = np.ma.MaskedArray(y, mask, fill_value=np.nan)
    #
    levels = MaxNLocator(nbins=300).tick_values(z.min(), z.max())
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # z = np.ma.array(z, mask=z < .01) # optionally mask to white below certain value
    cnt = ax.contourf(x, y, z,
            # levels=levels,
            # cmap=cmap,
            vmin=small,
        alpha=.3, antialiased=True)#, norm=LogNorm(vmin=z.min(), vmax=z.max()))
    
    
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
    
    cbar = plt.colorbar(cnt, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel(r'Absolute change in Hamiltonian, $|{1 - e^{-\delta H(p,x)}}|$')
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
    
    n_points = 300    # n**2 is the number of points
    x = np.linspace(-15, 15, n_points, endpoint=True)
    x,y = np.meshgrid(x, x)
    
    z = [np.exp(-model.sampler.potential.uE(Periodic_Lattice(np.asarray([i,j])))) \
              for i,j in zip(np.ravel(x), np.ravel(y))]
    z = np.asarray(z).reshape(n_points, n_points)
    return x,y,z
#
if __name__ == '__main__':
    
    n_burn_in = 25
    n_samples = 1000
    
    pot = Klein_Gordon(phi_4=-0.05*np.math.factorial(4),m=0.05)
    spacing = 0.1
    n, dim = 50, 1
    x0 = np.random.random((n,)*dim)
    x0 = np.asarray(x0, dtype=np.float64)
    x0 = Periodic_Lattice(x0)
    model = Model(x0, pot=pot, step_size=0.1, n_steps=20, rand_steps=True, spacing=spacing)
    
    # adjust for nice plotting
    print "Running Model: {}".format(__file__)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose=True)
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
