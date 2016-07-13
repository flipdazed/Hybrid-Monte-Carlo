import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from scipy import stats

from data import store
from correlations import acorr, corr
from models import Basic_GHMC as Model
from utils import saveOrDisplay, prll_map
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
    
    ax[0].set_xlabel(r'HMC trajectories between samples, $(\tau_{i+t} - \tau_{i}) / n\delta\tau$')
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
def main(x0, pot, file_name, 
        n_samples, n_burn_in, 
        mixing_angles, angle_labels, c_len=20, 
        step_size = .5, n_steps = 1, spacing = 1., 
        save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        mixing_angles :: iterable :: mixing angles for the HMC algorithm
        angle_labels :: list :: list of labels for the angles provided
    
    Optional Inputs
        c_len :: int :: length of autocorellation
        step_size :: float :: MDMC step size
        n_steps :: int :: number of MDMC steps
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    lines = {} # contains the label as the key and a tuple as (x,y) data in the entry
    acs = {}
    
    separations = range(c_len)
    opFn = lambda samples: corr.twoPoint(samples, separation=0)
    op_name = r'$\hat{O} = \langle x(0)x(0) \rangle$'
    
    rng = np.random.RandomState()
    
    subtitle = r"Potential: {}; Lattice Shape: ${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        pot.name, x0.shape, spacing, step_size, n_steps)
    length = x0.size
    
    print 'Running Model: {}'.format(file_name)
    def coreFunc(a):
        """runs the below for an angle, a"""
        i,a = a
        model = Model(x0, pot=pot, spacing=spacing, rng=rng, step_size = step_size,
          n_steps = n_steps)
        c = acorr.Autocorrelations_1d(model, 'run', 'samples')
        c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = a, verbose=True, verb_pos=i)
        ac = c.getAcorr(separations, opFn)
        
        av_corr = c.op_mean
        return av_corr, ac
    
    # use multiprocessing
    ans = prll_map(coreFunc, zip(range(len(mixing_angles)), mixing_angles), verbose=False)
    x0x0, ans = zip(*ans)
    print '\n'*len(mixing_angles) # hack to avoid overlapping!
    for av_corr,a in zip(x0x0, mixing_angles):
        print '> measured at angle:{:3.1f}: <x(0)x(0)> = {}'.format(a, av_corr)
    
    # create dictionary for plotting
    acs = {l:(separations, y) for y, l in zip(ans, angle_labels)}
    print 'Finished Running Model: {}'.format(file_name)
    
    store.store(lines, file_name, '_lines')
    store.store(acs, file_name, '_acs')
    plot(acs, lines,
        subtitle, op_name,
        save = saveOrDisplay(save, file_name))
    