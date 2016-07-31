import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import random
from scipy import stats

from data import store
from correlations import acorr, corr, errors
from models import Basic_GHMC as Model
from utils import saveOrDisplay, prll_map
from plotter import Pretty_Plotter, PLOT_LOC

# generatte basic colours list
clist = [i for i in colors.ColorConverter.colors if i != 'w']
colour = [i for i in random.sample(clist, len(clist))]

# generate only dark colours
darkclist = [i for i in colors.cnames if 'dark' in i]
darkcolour = [i for i in random.sample(darkclist, len(darkclist))]
lightcolour = map(lambda strng: strng.replace('dark',''), darkcolour)

theory_colours = iter(colour)
measured_colours = iter(colour)

def plot(acns, lines, subtitle, op_name, save):
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
    
    ax[0].set_xlabel(r'Fictitious sample time, '                            \
        + r'$t = \sum^j_{i=0}\tau_i \stackrel{j\to\infty}{=} j\bar{\tau} '  \
        + r'= j \delta \tau \bar{n}$')
    
    ax[0].set_ylabel(r'Normalised Autocorrelation, $C(t)$')# \
        # + r'(\hat{O}_{i+t} - \langle\hat{O}\rangle) \rangle}/{\langle \hat{O}_0^2 \rangle}$')
    
    for label, (x, y, e) in sorted(acns.iteritems()):
        c = next(measured_colours)
        # for some reason I occisionally need to add a fake plot
        # p2 = ax[0].add_patch(Rectangle((0, 0), 0, 0, fc=c, linewidth=0, alpha=.4, label=label))
        try:
            ax[0].fill_between(x, y-e, y+e, color=c, alpha=0.3, label=label)
        except Exception, e:
            print 'Failed for "{}"'.format(label)
            print 'Shapes: x:{} y:{} e:{}'.format(x.shape, y.shape, e.shape)
            print 'Error:'
            print e
        # ax[0].errorbar(x, y, yerr=e, c=c, ecolor='k', ms=3, fmt='o', alpha=0.5,
        #     label=label)
    
    for label, line in sorted(lines.iteritems()):
        dc = next(theory_colours)
        ax[0].plot(*line, linestyle='-', linewidth=1, alpha=1, label=label, color = dc)
    
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
        mixing_angles, angle_labels, opFn, op_name, separations, 
        rand_steps= False, step_size = .5, n_steps = 1, spacing = 1.,
        Theory_Cls = None,
        save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        mixing_angles :: iterable :: mixing angles for the HMC algorithm
        angle_labels  :: list :: list of labels for the angles provided
        opFn        :: func :: function for autocorellations - takes one input: samples
        op_name     :: str :: name of opFn for plotting
        separations :: iterable :: lengths of autocorellations
    
    Optional Inputs
        rand_steps :: bool :: probability of with prob
        step_size :: float :: MDMC step size
        n_steps :: int :: number of MDMC steps
        spacing :: float :: lattice spacing
        Theory_Cls :: class :: a class object with .eval() for evaluating a theory
        save :: bool :: True saves the plot, False prints to the screen
    
    Expectations
        1.  Theory_Cls must accept **kwargs: 'pa', 'tau' as it is expected it will vary
            with respect to these parameters
        2.  Theory_Cls must have a function '.eval(pa=pa, tau=tau)' to return predictions
    """
    
    # required declarations. If no lines are provided this still allows iteration for 0 times
    lines = {}  # contains the label as the key and a tuple as (x,y) data in the entry
    acs = {}
    
    if not isinstance(separations, np.ndarray): separations = np.asarray(separations)
    rng = np.random.RandomState()
    
    subtitle = r"Potential: {}; Lattice: ${}$; $a={:.1f}; \delta\tau={:.2f}; n={}$".format(
        pot.name, x0.shape, spacing, step_size, n_steps)
    
    print 'Running Model: {}'.format(file_name)
    
    def coreFunc(a):
        """runs the below for an angle, a
        Allows multiprocessing across a range of values for a
        """
        i,a = a
        model = Model(x0, pot=pot, spacing=spacing, rng=rng, step_size = step_size,
          n_steps = n_steps, rand_steps=rand_steps)
        
        c = acorr.Autocorrelations_1d(model)
        c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = a, verbose=True, verb_pos=i)
        
        acs = c.getAcorr(separations, opFn, norm = False)   # non norm for uWerr
        ans = errors.uWerr(c.op_samples, acorr=acs)         # get errors
        _, _, _, itau, itau_diff, _, acns = ans             # extract data
        w = errors.getW(itau, itau_diff, n=n_samples)       # get window length
        acns_err = errors.acorrnErr(acns, w, n_samples)     # get autocorr errors
        
        # get parameters generated
        traj = c.trajs
        p = c.model.p_acc
        xx = c.op_mean
        return xx, acns, acns_err, p, w
    #
    print 'Finished Running Model: {}'.format(file_name)
    # use multiprocessing
    
    al = len(mixing_angles) # the number of mixing angles
    if al == 1:             # don't use multiprocessing for just 1 mixing angle
        a = mixing_angles[0]
        ans = [coreFunc((i, a)) for i,a in enumerate(mixing_angles)]
    else:                   # use multiprocessing for a number of mixing angles
        ans = prll_map(coreFunc, zip(range(al), mixing_angles), verbose=False)
    
    # results have now been obtained. This operation is a dimension shuffle
    # Currently the first iterable dimension contains one copy of each item
    # Want to split into separate arrays each of length n
    xx, acns, acns_err, ps, ws = zip(*ans)
    
    print '\n'*al           # hack to avoid overlapping with the progress bar from multiprocessing
    out = lambda p,x,a: '> measured at angle:{:3.1f}: <x(0)x(0)> = {}; <P_acc> = {:4.2f}'.format(a,x,p)
    for p, x, a in zip(ps, xx, mixing_angles):
        print out(p,x,a)    # print output as defined above
    
    # Decide a good total length for the plot
    w = np.max(ws)                                  # same length for all theory and measured data
    
    # Create Dictionary for Plotting Measured Data
    windowed_separations = separations[:2*w]        # cut short to 2*window for a nicer plot
    x = windowed_separations*step_size*n_steps      # create the x-axis in "ficticious HMC-time"
    aclabel = r'Measured: $C_{\phi^2}(t; '             \
        + r'\bar{P}_{\text{acc}}'+r'={:4.2f}; '.format(p)
    yelpw = zip(acns, acns_err, angle_labels, ps, ws)   # this is an iterable of all a/c plot values
    yelpw = yelpw                                       # cut to the same window length as x-axis
    
    # create the dictionary item to pass to plot()
    acns = {aclabel+r'\theta = {})$'.format(l) :(x, y[:2*w], e[:2*w]) for y,e,l,p,w_i in yelpw}
    
    if Theory_Cls is not None: # Create Dictionary for Plotting Theory
        m = Theory_Cls                                      # create a shortcut for theory class
        fx_res = 100                                        # points per x-value
        fx_points = 2*w*fx_res+1                            # number of points to use
        fx_f = windowed_separations[-1]*step_size*n_steps   # last trajectory separation length to plot
        fx = np.linspace(0, fx_f, fx_points, True)          # create the x-axis for the theory
        windowed_ps = ps[:2*w]                              # windowed acceptance probabilities
        # calculcate theory across all tau, varying p_acc and normalise
        vFn = lambda pt: m.eval(t=fx, pa=pt[0], theta=pt[1])/m.eval(t=0,pa=pt[0], theta=pt[1])
        fs = map(vFn, zip(ps, mixing_angles))               # map the a/c function to acceptance & angles
        th_label = r'Theory: $C_{\phi^2}(t; ' \
            + r'\bar{P}_{\text{acc}} \approx 1; '
        fl = zip(fs, angle_labels)                          # this is an iterable of all theory plot values
        fl = fl[:2*w]                                       # cut to the same window length as x-axis
        
        # create the dictionary item to pass to plot()
        lines = {th_label+ r'\theta = {})$'.format(l): (fx, f) for f,l in fl if f is not None} 
    else:
        pass # lines = {} has been declared at the top so will allow the iteration in plot()
    
    # Bundle all data ready for Plot() and store data as .pkl or .json for future use
    all_plot = {'acns':acns, 'lines':lines, 'subtitle':subtitle, 'op_name':op_name}
    store.store(all_plot, file_name, '_allPlot')
    
    plot(acns, lines, subtitle, op_name,
        save = saveOrDisplay(save, file_name))
    
