import os
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis
from matplotlib.colors import LogNorm

from common.hmc.lattice import Model
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

def plot(x, y, z, save = 'dynamics_qho_constEn_2d.png'):
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    fig.suptitle(r'Energy Drift as a function of Integrator Parameters', 
        fontsize=pp.ttfont)
    name = 'QHO'
    ax[0].set_title(r'Potential: {}'.format(name),
        fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Number of Integrator Steps, $n$')
    ax[0].set_ylabel(r'Integrator Step Size, $\epsilon$')
    
    z = np.asarray(z).reshape(x.size, y.size)
    
    p = ax[0].contourf(x, y, z, 200,
        norm=LogNorm(vmin=z.min(), vmax=z.max()))
    
    # add colorbar and label
    cbar = plt.colorbar(p, ax=ax[0], shrink=0.9)
    cbar.ax.set_ylabel(r'Absolute change in Hamiltonian, $|{1 - e^{-\delta H(p,x)}}|$')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def dynamicalEnergyChange(pot, step_sample, step_sizes):
    """Iterates the dynamics for the steps and the step sizes
    returns the absolute change in the hamiltonian for each
    parameter configuration
    
    Required Inputs
        pot         :: potential class - see hmc.potentials
        step_sample :: int   :: sample array of integrator step lengths
        step_sizes  :: float :: sample array of integrator step sizes
    """
    
    model = Model(pot = pot)
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p
    x0 = model.x0
    # calculate original hamiltonian and set starting vals
    h_old = model.pot.hamiltonian(p0, x0)
    
    # set up a mesh grid of the steps and sizes
    step_sample, step_sizes = np.meshgrid(step_sample, step_sizes)
    
    diffs = []
    for n_steps_i, step_size_i in zip(np.ravel(step_sample), np.ravel(step_sizes)):
        
        # set new parameters
        model.dynamics.n_steps = n_steps_i
        model.dynamics.step_size = step_size_i
        
        # obtain new duynamics and resultant hamiltonian
        model.dynamics.newPaths()
        pf, xf = model.dynamics.integrate(copy(p0), copy(x0))
        h_new = model.pot.hamiltonian(pf, xf)
        
        bench_mark = np.exp(-(h_old-h_new))
        
        diff = np.abs(1. - bench_mark) # set new diff
        
        diffs.append(diff) # append to list for plotting
    
    return diffs
#
if '__main__' == __name__:
    
    step_sizes  = [.001, .5]
    steps       = [1, 500]
    n_steps     = 25
    n_sizes     = 25
    
    steps = np.linspace(steps[0], steps[1], n_steps, True, dtype=int)
    step_sizes = np.linspace(step_sizes[0], step_sizes[1], n_sizes, True)
        
    print 'Running Model'
    print 'Running Model'
    pot = QHO()
    en_diffs = dynamicalEnergyChange(pot, steps, step_sizes)
    print 'Finished Running Model'
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    plot(x = steps, y = step_sizes, z = en_diffs,
        save = save_name,
        # save = False
        )