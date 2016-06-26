
import os
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from hmc import checks
from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis

from common.hmc.lattice import Model
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

def plot(y1, y2, all_lines = False, save = 'dynamics_qho_constEn_1d.png'):
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = r"\usepackage{amsmath}"
    pp._updateRC()
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    fig.suptitle(r"Components of Hamiltonian, $H'$, during Leap Frog integration",
        fontsize=pp.ttfont)
    
    name  = "QHO"
    ax[0].set_title(r'Potential: {}'.format(name),
        fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Number of Integrator Steps, $n$')
    ax[0].set_ylabel(r'Energy')
    
    steps = np.linspace(0, y1.size, y1.size, True)
    action, k, u = zip(*y2)
    
    action = np.asarray(action)
    k = np.asarray(k)
    u = np.asarray(u)
    
    h = ax[0].plot(steps, y1+np.asarray(action), # Full Hamiltonian
        label=r"$H' = T(\pi) + S(x,t)$", color='blue',
        linewidth=5., linestyle = '-', alpha=0.2)
    
    if all_lines:
        t = ax[0].plot(steps, np.asarray(y1), # Kinetic Energy (conjugate)
            label=r'$T(\pi)$', color='darkred',
            linewidth=3., linestyle='-', alpha=0.2)
        
        s = ax[0].plot(steps, np.asarray(action), # Potential Energy (Action)
            label=r'$S(x,t) = \sum_{n} (T_S + V_S)$', color='darkgreen',
            linewidth=3., linestyle='-', alpha=0.2)
        
        t_s = ax[0].plot(steps, np.asarray(k),  # Kinetic Energy in Action
            label=r'$\sum_{n} T_S$', color='red',
            linewidth=1., linestyle='--', alpha=1.)
        
        v_s = ax[0].plot(steps, np.asarray(u),  # Potential Energy in Action
            label=r'$\sum_{n} V_S$', color='green',
            linewidth=1., linestyle='--', alpha=1.)
    
    # add legend
    ax[0].legend(loc='upper left', shadow=True, fontsize = pp.axfont)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def dynamicalEnergyChange(pot, n_steps, step_size):
    """Iterates the dynamics for the steps and the step sizes
    returns the absolute change in the hamiltonian for each
    parameter configuration
    
    Required Inputs
        pot         :: potential class - see hmc.potentials
        step_sample :: int   :: sample array of integrator step lengths
        step_sizes  :: float :: sample array of integrator step sizes
    """
    
    model = Model(
        pot       = pot,
        n_steps   = n_steps, 
        step_size = step_size,
        spacing   = 1.
        )
    model.dynamics.save_path = True
    model.pot.debug = True
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p
    x0 = model.x0
    
    # calculate original hamiltonian and set starting vals
    h0    = model.pot.hamiltonian(p0, x0)
    kE0   = model.pot.kE(p0)
    uE0   = model.pot.uE(x0)
    
    # Setting debug = True returns a tuple
    # from the potential: (action, action_ke, action_ue)
    if len(list(uE0)) > 1: # list() as can be np.array or plain float
        check_uE0 = uE0[0] # if a debug then this is action
    else:
        check_uE0 = uE0    # if not debug then as normal
    
    # obtain new duynamics and resultant hamiltonian
    model.dynamics.newPaths()
    pf, xf = model.dynamics.integrate(copy(p0), copy(x0))
    
    kE_path = [model.pot.kE(i) for i in model.dynamics.p_ar]
    uE_path = []
    
    for i in model.dynamics.x_ar:
        x0.get = i      # must pass as Periodic_Lattice() instance
        uE_path.append(model.pot.uE(x0))
    
    kins = np.asarray([kE0] + kE_path)
    pots = [uE0] + uE_path

    return kins, pots 
#
if '__main__' == __name__:
    
    n_steps   = 500
    step_size = .01
        
    print 'Running Model'
    pot = QHO()
    kins, pots = dynamicalEnergyChange(pot, n_steps, step_size)
    print 'Finished Running Model'
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    print save_name
    plot(y1 = kins, y2 = pots, all_lines=True,
        save = save_name
        # save = False
        )