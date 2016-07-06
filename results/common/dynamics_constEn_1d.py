import numpy as np
import matplotlib.pyplot as plt

from models import Basic_HMC as Model
from utils import saveOrDisplay

from plotter import Pretty_Plotter, PLOT_LOC

def plot(y1, y2, subtitle, save, all_lines = False):
    """A plot of y1 and y2 as functions of the steps which
    are implicit from the length of the arrays
    
    Required Inputs
        y1 :: np.array :: conj kinetic energy array
        y2   :: np.array :: shape is either (n, 3) or (n,)
        subtitle :: string :: subtitle for the plot
        save :: bool :: True saves the plot, False prints to the screen
    
    Optional Inputs
        all_lines :: bool :: if True, plots hamiltonian as well as all its components
    """
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = r"\usepackage{amsmath}"
    pp._updateRC()
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    fig.suptitle(r"Change in Energy during Leap Frog integration",
        fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Number of Leap Frog Steps, $n$')
    ax[0].set_ylabel(r'Change in Energy, $\delta E = E_n - E_0$')
    
    steps = np.linspace(0, y1.size, y1.size, True)
    
    # check for multiple values in the potential
    multi_pot = (y2.shape[-1] > 1)
    if multi_pot:
        action, k, u = zip(*y2)
        k = np.asarray(k)
        u = np.asarray(u)
    else:
        action = y2
        
    action = np.asarray(action)
    
    h = ax[0].plot(steps, y1+np.asarray(action), # Full Hamiltonian
        label=r"$\delta H = \delta T(\pi) + \delta S(x,t)$", color='blue',
        linewidth=5., linestyle = '-', alpha=0.2)
    
    if all_lines:
        t = ax[0].plot(steps, np.asarray(y1), # Kinetic Energy (conjugate)
            label=r'$\delta T(\pi)$', color='darkred',
            linewidth=3., linestyle='-', alpha=0.2)
        
        if multi_pot:
            s = ax[0].plot(steps, np.asarray(action), # Potential Energy (Action)
                label=r'$\delta S(x,t) = \sum_{n} (\delta T_S + \delta V_S)$', color='darkgreen',
                linewidth=3., linestyle='-', alpha=0.2)
                
            t_s = ax[0].plot(steps, np.asarray(k),  # Kinetic Energy in Action
                label=r'$\sum_{n} \delta T_S$', color='red',
                linewidth=1., linestyle='--', alpha=1.)
            
            v_s = ax[0].plot(steps, np.asarray(u),  # Potential Energy in Action
                label=r'$\sum_{n} \delta V_S$', color='green',
                linewidth=1., linestyle='--', alpha=1.)
        else:
            s = ax[0].plot(steps, np.asarray(action), # Potential Energy (Action)
                label=r'$\delta S(x,t) = \frac{1}{2}\delta(x^2)$', color='darkgreen',
                linewidth=3., linestyle='-', alpha=0.2)
    
    # add legend
    ax[0].legend(loc='upper left', shadow=True, fontsize = pp.axfont)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def dynamicalEnergyChange(x0, pot, n_steps, step_size):
    """Iterates the dynamics for the steps and the step sizes
    returns the absolute change in the hamiltonian for each
    parameter configuration
    
    Required Inputs
        x0          :: dependent on the potential used
        pot         :: potential class - see hmc.potentials
        step_sample :: int   :: sample array of integrator step lengths
        step_sizes  :: float :: sample array of integrator step sizes
    """
    rng = np.random.RandomState()
    model = Model(x0, pot,
        n_steps   = n_steps, 
        step_size = step_size,
        rng=rng)
    model.dynamics.save_path = True # saves the dynamics path
    model.pot.debug = True          # saves all energies
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p0
    x0 = model.sampler.x0
    
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
    pf, xf = model.dynamics.integrate(p0.copy(), x0.copy(), verbose=True)
    
    kE_path = [model.pot.kE(i) for i in model.dynamics.p_ar]
    uE_path = [model.pot.uE(i) for i in model.dynamics.x_ar]
    
    kins = np.asarray([kE0] + kE_path) - kE0
    pots = np.asarray([uE0] + uE_path) - uE0
    return kins, pots 
#
def main(x0, pot, file_name, save = False, n_steps   = 500, step_size = .1, all_lines = True):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
    
    Optional Inputs
        save :: bool :: True saves the plot, False prints to the screen
        n_steps :: list :: LF trajectory lengths
        step_size :: float :: Leap Frog step size
        all_lines :: bool :: if True, plots hamiltonian as well as all its components
    """
    print 'Running Model: {}'.format(file_name)
    kins, pots = dynamicalEnergyChange(x0, pot, n_steps, step_size)
    print 'Finished Running Model: {}'.format(file_name)
    
    plot(y1 = kins, y2 = pots, all_lines = all_lines,
        subtitle = r'Potential: {}, Lattice shape: {}'.format(pot.name, x0.shape),
        save = saveOrDisplay(save, file_name)
        )
    pass