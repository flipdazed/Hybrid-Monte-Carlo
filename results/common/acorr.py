from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import random
from scipy import stats

from data import store
from utils import saveOrDisplay, prll_map

from correlations import acorr, errors
from models import Basic_GHMC as Model
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
    pp.params['text.latex.preamble'] =r"\usepackage{bbold}"
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
            if e is not None:
                # ax[0].fill_between(x, y-e, y+e, color=c, alpha=0.3, label=label)
                ax[0].errorbar(x, y, yerr=e, c=c, ecolor='k', ms=3, fmt='o', alpha=0.5,
                    label=label)
            else:
                ax[0].scatter(x, y, c=c, ms=3, fmt='o', alpha=0.5, label=label)
        except Exception, e:
            print '\nFailed for "{}"'.format(label)
            print 'Wrong types? x:{} y:{} e:{}'.format(type(x), type(y), type(e))
            print '\nError:'
            print e
    
    for label, line in sorted(lines.iteritems()):
        dc = next(theory_colours)
        ax[0].plot(*line, linestyle='-', linewidth=1, alpha=1, label=label, color = dc)
    
    xi,xf = ax[0].get_xlim()
    ax[0].set_xlim(xmin=xi-0.05*(xf-xi)) # give a decent view of the first point
    yshft = np.diff(ax[0].get_ylim())
    ax[0].set_ylim(ymax=1 + .05*yshft, ymin=0-.05*yshft) # give 5% extra room at top
#    ax[0].set_yscale("log", nonposy='clip')
    
    ax[0].legend(loc='best', shadow=True, fontsize = pp.axfont)
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name,
        n_samples, n_burn_in,
        mixing_angles, angle_labels, opFn, op_name, separations, 
        rand_steps= False, step_size = .5, n_steps = 1, spacing = 1.,
        acFunc = None,
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
        acFunc :: func :: function for evaluating autocorrelations
        save :: bool :: True saves the plot, False prints to the screen
    """
    al = len(mixing_angles) # the number of mixing angles
    send_prll = prll_map if al==1 else None
    
    # required declarations. If no lines are provided this still allows iteration for 0 times
    lines = {}  # contains the label as the key and a tuple as (x,y) data in the entry
    acs = {}
    
    if not isinstance(separations, np.ndarray): separations = np.asarray(separations)
    rng = np.random.RandomState()
    
    subtitle = r"Potential: {}; Lattice: ${}$; $a={:.1f}; \delta\tau={:.2f}; n={}; m={:.1f}$".format(
        pot.name, x0.shape, spacing, step_size, n_steps, pot.m)
    
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
        store.store(c.model.samples, file_name, '_samples')
        store.store(c.model.traj, file_name, '_trajs')
        acs = c.getAcorr(separations, opFn, norm = False, prll_map=send_prll)   # non norm for uWerr
        
        # get parameters generated
        traj        = np.cumsum(c.trajs)
        p           = c.model.p_acc
        xx          = c.op_mean
        acorr_seps  = c.acorr_ficticous_time
        acorr_counts= c.acorr_counts
        store.store(p, file_name, '_probs')
        store.store(acs, file_name, '_acs')
        
        ans = errors.uWerr(c.op_samples, acorr=acs)         # get errors
        _, _, _, itau, itau_diff, _, acns = ans             # extract data
        w = errors.getW(itau, itau_diff, n=n_samples)       # get window length
        acns_err = errors.acorrnErr(acns, w, n_samples)     # get error estimates
        
        if rand_steps: # correct for N being different for each correlation
            acns_err *= np.sqrt(n_samples)/acorr_counts
            
        
        return xx, acns, acns_err, p, w, traj, acorr_seps
    #
    print 'Finished Running Model: {}'.format(file_name)
    # use multiprocessing
    
    if al == 1:             # don't use multiprocessing for just 1 mixing angle
        a = mixing_angles[0]
        ans = [coreFunc((i, a)) for i,a in enumerate(mixing_angles)]
    else:                   # use multiprocessing for a number of mixing angles
        ans = prll_map(coreFunc, zip(range(al), mixing_angles), verbose=False)
    
    # results have now been obtained. This operation is a dimension shuffle
    # Currently the first iterable dimension contains one copy of each item
    # Want to split into separate arrays each of length n
    xx, acns, acns_err, ps, ws, ts, acxs = zip(*ans)
    
    print '\n'*al           # hack to avoid overlapping with the progress bar from multiprocessing
    out = lambda p,x,a: '> measured at angle:{:3.1f}: <x(0)x(0)> = {}; <P_acc> = {:4.2f}'.format(a,x,p)
    for p, x, a in zip(ps, xx, mixing_angles):
        print out(p,x,a)    # print output as defined above
    
    # Decide a good total length for the plot
    w = np.max(ws)                                  # same length for all theory and measured data
    print 'Window is:{}'.format(w)
    if np.isnan(w): 
        alen = int(len(separations)/2)
    else:
        alen = 2*w
    
    # Create Dictionary for Plotting Measured Data
    aclabel = r'Measured: $C_{\phi^2}(t; '             \
        + r'\bar{P}_{\text{acc}}'+r'={:4.2f}; '.format(p)
    yelpwx = zip(acns, acns_err, angle_labels, ps, ws, acxs)  # this is an iterable of all a/c plot values
    
    # create the dictionary item to pass to plot()
    acns = {aclabel+r'\theta = {})$'.format(l) :(x[:alen], y[:alen], e[:alen]) for y,e,l,p,w_i,x in yelpwx}
    
    if acFunc is not None: # Create Dictionary for Plotting Theory
        fx_f = np.max(np.asarray([a[:alen] for a in acxs]))  # last trajectory separation length to plot
        fx_res = step_size*0.1                              # points per x-value
        fx_points = fx_f/fx_res+1                           # number of points to use
        fx = np.linspace(0, fx_f, fx_points, True)          # create the x-axis for the theory
        windowed_ps = ps[:alen]                              # windowed acceptance probabilities
        # calculcate theory across all tau, varying p_acc and normalise
        normFn = lambda pt: np.array([acFunc(t=xi, pa=pt[0], theta=pt[1]) for xi in fx]) / acFunc(t=0, pa=pt[0], theta=pt[1])
        fs = map(normFn, zip(ps, mixing_angles))            # map the a/c function to acceptance & angles
        th_label = r'Theory: $C_{\phi^2}(t; \bar{P}_{\text{acc}} = '
        pfl = zip(ps, fs, angle_labels)                     # this is an iterable of all theory plot values
        pfl = pfl[:alen]                                     # cut to the same window length as x-axis
        
        # create the dictionary item to pass to plot()
        lines = {th_label+ r'{:4.2f}; \theta = {})$'.format(p, l): (fx, f) for p,f,l in pfl if f is not None} 
    else:
        pass # lines = {} has been declared at the top so will allow the iteration in plot()
    
    # Bundle all data ready for Plot() and store data as .pkl or .json for future use
    all_plot = {'acns':acns, 'lines':lines, 'subtitle':subtitle, 'op_name':op_name}
    store.store(all_plot, file_name, '_allPlot')
    
    plot(acns, lines, subtitle, op_name,
        save = saveOrDisplay(save, file_name))
    
