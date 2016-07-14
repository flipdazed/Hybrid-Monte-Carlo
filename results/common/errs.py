import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

from data import store
from correlations import acorr, corr, errors
from models import Basic_GHMC as Model
from utils import saveOrDisplay
from plotter import Pretty_Plotter, PLOT_LOC

def plot(acs, lines, subtitle, op_name, save):
    """Plots the two-point correlation function
    
    Required Inputs
        acs              :: {label:(x,y)} :: plots (x,y) as a stem plot with label=label
        lines            :: {label:(x,y)} :: plots (x,y) as a line with label=label
        subtitle :: str  :: subtitle for the plot
        op_name  :: str  :: the name of the operator for the title
        save     :: bool :: True saves the plot, False prints to the screen
    
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
    
    fig.suptitle(r"Autocorrelation Function for {}".format(op_name),
        fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Av. trajectories between samples, $\langle\tau_{i+t} - \tau_{i}\rangle / n\delta\tau$')
    ax[0].set_ylabel(r'$\mathcal{C}(t) = {\langle (\hat{O}_i - \langle\hat{O}\rangle)' \
        + r'(\hat{O}_{i+t} - \langle\hat{O}\rangle) \rangle}/{\langle \hat{O}_0^2 \rangle}$')
    
    clist = [i for i in colors.ColorConverter.colors if i != 'w']
    colour = (i for i in random.sample(clist, len(clist)))
    for label, line in lines.iteritems():
        ax[0].plot(*line, linestyle='-', linewidth=2., alpha=0.4, label=label, color = next(colour))
    
    for label, line in acs.iteritems():
        ax[0].plot(*line, linestyle='-', linewidth=2., alpha=0.4, label=label, color = next(colour))
    
    # for label, stem in stems.iteritems():
    #     ax[0].stem(*stem, markerfmt='o', linefmt='k:', basefmt='k-',label=label)
    
    xi,xf = ax[0].get_xlim()
    ax[0].set_xlim(xmin=xi-0.05*(xf-xi)) # give a decent view of the first point
    ax[0].set_ylim(ymax=1 + .05*np.diff(ax[0].get_ylim())) # give 5% extra room at top
#    ax[0].set_yscale("log", nonposy='clip')
    
    ax[0].legend(loc='best', shadow=True, fontsize = pp.axfont)
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, mixing_angle, opFn,
        rand_steps = False, step_size = .5, n_steps = 1, spacing = 1., 
        save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        mixing_angle :: iterable :: mixing angles for the HMC algorithm
        opFn        :: func :: a function o run over samples
    
    Optional Inputs
        rand_steps :: bool :: probability of with prob
        step_size :: float :: MDMC step size
        n_steps :: int :: number of MDMC steps
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    
    lines = {} # contains the label as the key and a tuple as (x,y) data in the entry
    acs = {}
    
    rng = np.random.RandomState()
    
    subtitle = r"Potential: {}; Lattice Shape: ${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        pot.name, x0.shape, spacing, step_size, n_steps)
    
    print 'Running Model: {}'.format(file_name)
    model = Model(x0, pot=pot, spacing=spacing, rng=rng, step_size = step_size,
      n_steps = n_steps, rand_steps=rand_steps)
    c = acorr.Autocorrelations_1d(model)
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = mixing_angle, verbose=True)
    cfn = opFn(c.model.samples)
    print 'Finished Running Model: {}'.format(file_name)
    
    # get parameters generated
    traj = c.model.traj
    p = c.model.p_acc
    xx = np.average(cfn)
    
    print '> measured at angle:{:3.1f}: <x^2> = {}; <P_acc> = {:4.2f}'.format(mixing_angle, xx , p)
    
    store.store(cfn, file_name, '_cfn')
    ans = errors.uWerr(cfn)
    f_aav, f_diff, f_ddiff, itau, itau_diff, itau_aav = ans