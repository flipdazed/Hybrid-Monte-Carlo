import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from models import Basic_HMC as Model
from utils import saveOrDisplay, prll_map

from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis

#
def plot(x, y, z, subtitle, save, log_scale=False, **kwargs):
    """plots a contour plot of z(x,y)
    
    Required Inputs
        x,y :: np.array :: must be same lengths and 1D
        z   :: np.array :: must be length x.size * y.size
        subtitle :: string :: subtitle for the plot
        save :: bool :: True saves the plot, False prints to the screen
    
    Optional Inputs
        log_scale :: bool :: puts a log_scaling on the contour plot if True
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    fig.suptitle(r"Change in Energy during integration varying with Leap-Frog parameters",
        fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Number of Leap Frog Steps, $n$')
    ax[0].set_ylabel(r'Leap Frog Step Size, $\epsilon$')
    
    z = np.asarray(z).reshape(x.size, y.size)
    
    if log_scale:
        p = ax[0].contourf(x, y, z, 200,
            norm=LogNorm(vmin=z.min(), vmax=z.max()))
    else:
        p = ax[0].contourf(x, y, z, 200)
    
    # add colorbar and label
    cbar = plt.colorbar(p, ax=ax[0], shrink=0.9)
    cbar.ax.set_ylabel(r'Absolute change in Hamiltonian, $|{1 - e^{-\delta H(p,x)}}|$')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def dynamicalEnergyChange(x0, pot, step_sample, step_sizes):
    """Iterates the dynamics for the steps and the step sizes
    returns the absolute change in the hamiltonian for each
    parameter configuration
    
    Required Inputs
        x0          :: dependent on the potential used
        pot         :: potential class - see hmc.potentials
        step_sample :: int   :: sample array of integrator step lengths
        step_sizes  :: float :: sample array of integrator step sizes
    """
    
    model = Model(x0, pot = pot)
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p0
    x0 = model.sampler.x0
    
    # calculate original hamiltonian and set starting vals
    h_old = model.pot.hamiltonian(p0, x0)
    
    # set up a mesh grid of the steps and sizes
    step_sample, step_sizes = np.meshgrid(step_sample, step_sizes)
    steps_sizes_grid = zip(np.ravel(step_sample), np.ravel(step_sizes))
    
    def coreFunc(step_sizes_grid):
        """function for multiprocessing support"""
        
        # set new parameters
        n_steps_i, step_size_i = step_sizes_grid
        model.dynamics.n_steps = n_steps_i
        model.dynamics.step_size = step_size_i
        
        # obtain new duynamics and resultant hamiltonian
        model.dynamics.newPaths()
        pf, xf = model.dynamics.integrate(p0.copy(), x0.copy())
        h_new = model.pot.hamiltonian(pf, xf)
        
        bench_mark = np.exp(-(h_old-h_new))
        
        diff = np.abs(1. - bench_mark) # set new diff
        return diff
    
    # use multi-core support to speed up
    diffs = prll_map(coreFunc, steps_sizes_grid, verbose=True)
    
    return diffs
#
def main(x0, pot, file_name, save = False, step_sizes = [.001, .1], steps = [1, 100], n_steps = 10, n_sizes = 10, **kwargs):
    """A wrapper for all potentials
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
    
    Optional Inputs
        save :: bool :: True saves the plot, False prints to the screen
        step_sizes :: list :: a list with 2 values: [min, max] step sizes
        steps :: list :: a list with 2 values: [min, max] steps (LF trajectory lengths)
        n_steps :: int :: the resolution (number) of step lengths to sample
        n_sizes :: int :: the resolution (number) of step sizes to sample
        **kwargs :: keyword arguments :: The only one that is any use is log_scale = True|False
                                         log_scale will put a log_scaling on the contour plot
    Expectations
        x0 matches the dimensions allowed by the potential used
    """
    steps = np.linspace(steps[0], steps[1], n_steps, True, dtype=int)
    step_sizes = np.linspace(step_sizes[0], step_sizes[1], n_sizes, True)
        
    print 'Running Model: {}'.format(file_name)
    en_diffs = dynamicalEnergyChange(x0, pot, steps, step_sizes)
    print 'Finished Running Model: {}'.format(file_name)
    
    plot(x = steps, y = step_sizes, z = en_diffs,
        subtitle = r'Potential: {}, Lattice shape: {}'.format(pot.name, x0.shape),
        save = saveOrDisplay(save, file_name),
        **kwargs
        )
    pass