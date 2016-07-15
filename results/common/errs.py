import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

from data import store
from correlations import acorr, corr, errors
from models import Basic_GHMC as Model
from utils import saveOrDisplay
from plotter import Pretty_Plotter, PLOT_LOC

from scipy.optimize import curve_fit
def expFit(t, a, b, c):
    return a + b * np.exp(-t / c)

def plot(lines, w, subtitle, labels, op_name, save):
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
    
    fig, ax = plt.subplots(3, sharex=True, figsize = (8, 8))
    fig.suptitle(r"Autocorrelation and Errors for {}".format(op_name),
        fontsize=pp.ttfont)
    
    fig.subplots_adjust(hspace=0.1)
    
    ### Add top pseudo-title and bottom shared x-axis label
    ax[0].set_title(subtitle, fontsize=pp.tfont)
    ax[-1].set_xlabel(r'Av. trajectories between samples, '\
        + r'$t = \langle \tau_{i+t} - \tau_{i} \rangle / n\delta\tau$')
    
    ### add the lines to the plots
    
    clist = [i for i in colors.ColorConverter.colors if i not in ['w', 'k']]
    colour = (i for i in random.sample(clist, len(clist)))
    
    ### add the Window stop point
    for a in range(1,len(ax)):            # iterate over each axis
        ax[a].axvline(x=w, linewidth=4, color='red', alpha=0.1)
    
    ax[0].set_ylabel(r'$g(w)$')
    x, y = lines[0]
    yp = y.copy() ; yp[yp < 0] = np.nan
    ym = y.copy() ; ym[ym >= 0] = np.nan
    ax[0].scatter(x, yp, marker='o', color='g', linewidth=2., alpha=0.6, label=r'$g(t) \ge 0$')
    ax[0].scatter(x, ym, marker='s', color='r', linewidth=2., alpha=0.6, label=r'$g(t) < 0$')
    
    ax[1].set_ylabel(r'$\tau_{\text{int}}(w)$')
    x, y, e = lines[1]
    ax[1].errorbar(x, y, yerr=e, label = r'MCMC data',
        markersize=5, color=next(colour), fmt='o', alpha=0.4, ecolor='k')
    
    ax[2].set_ylabel(r'Autocorrelation, $\Gamma(t)$')
    x, y, e = lines[2]
    ax[2].errorbar(x, y, yerr=e, label = r'MCMC data',
        markersize=5, color=next(colour), fmt='o', alpha=0.4, ecolor='k')
    
    for i in range(1, len(lines)):
        x, y = lines[i][:2]
        popt, pcov = curve_fit(expFit, x, y)
        l_th = r'Fit: $f(t) = {:.1f} + {:.1f}'.format(popt[0], popt[1]) \
            + r'e^{-t/' +'{:.2f}'.format(popt[2]) + r'}$'
        ax[i].plot(x, expFit(x, *popt), label = l_th,
            linestyle = '--', color='k', linewidth=1., alpha=0.6)
    
    xi,xf = ax[2].get_xlim()
    ax[2].set_xlim(xmin= xi-.05*(xf-xi)) # give a decent view of the first point
    for a in ax:
        yi,yf = a.get_ylim()
        a.set_ylim(ymax= yf+.05*(yf-yi), ymin= yi-.05*(yf-yi)) # give 5% extra room at top
        a.legend(loc='best', shadow=True, fontsize = pp.axfont)
    
    ### adds labels to the plots
    for i, text in labels.iteritems(): pp.add_label(ax[i], text, fontsize=pp.tfont)
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, mixing_angle, opFn,
        rand_steps = False, step_size = .5, n_steps = 1, spacing = 1., 
        save = False):
    """Takes a function: opFn. Runs HMC-MCMC. Runs opFn on HMC samples.
        Calculates Autocorrelation + Errors on opFn.
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: final plot will be saved with a similar name if save=True
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
    op_name = r'$\langle \phi^2 \rangle_{L}$'
    rng = np.random.RandomState()
    
    print 'Running Model: {}'.format(file_name)
    model = Model(x0, pot=pot, spacing=spacing, # set up model
        rng=rng, step_size = step_size,
        n_steps = n_steps, rand_steps=rand_steps)
    
    c = acorr.Autocorrelations_1d(model)                    # set up autocorrs
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in,    # run MCMC
        mixing_angle = mixing_angle, verbose=True)
    cfn = opFn(c.model.samples)                             # run func on HMC samples
    
    # get parameters generated
    traj = c.model.traj         # get trajectory lengths for each LF step
    p = c.model.p_acc           # get acceptance rates at each M-H step
    xx = np.average(cfn)        # get average of the function run over the samples
    print 'Finished Running Model: {}'.format(file_name)    
    
    store.store(cfn, file_name, '_cfn')     # store the function
    ans = errors.uWerr(cfn)                 # get the errors from uWerr
    
    print '> measured at angle:{:3.1f}:'.format(mixing_angle) \
        + ' <x^2>_L = {}; <P_acc>_HMC = {:4.2f}'.format(xx , p)
    
    subtitle = r"Potential: {}; Lattice Shape: ".format(pot.name) \
        + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
            x0.shape, spacing, step_size, n_steps)
    
    lines, labels, w = preparePlot(cfn, ans=ans, n = n_samples)
    
    plot(lines, w, subtitle,
        labels=labels, op_name = op_name,
        save = saveOrDisplay(save, file_name))
    pass
#
def preparePlot(op_samples, ans, n):
    """Prepares the plot according to the output of uWerr
    
    Required Inputs
        op_samples :: np.ndarray :: function acted upon the HMC samples
        ans :: tuple :: output form uWerr
        n   :: int   :: number of samples from MCMC
    """
    
    f_aav, f_diff, f_ddiff, itau, itau_diff, itaus, acn = ans
    
    print ' > Re-deriving error parameters...'
    t_max = int(n//2)
    fn = lambda t: acorr.acorr(op_samples=op_samples, mean=f_aav, separation=t, norm=None)
    ac  = np.asarray([fn(t=t) for t in range(0, t_max)])    # calculate autocorrelations
    w = round((f_ddiff/f_diff)**2*n - .5, 0)                # obtain the best windowing point
    itaus_diff  = errors.itauErrors(itaus, n=n)             # calcualte error in itau
    g_int = np.cumsum(ac[1:t_max]/ac[0])                                    # recreate the g_int function
    g = np.asarray([errors.gW(t, v, 1.5, n) for t,v in enumerate(g_int,1)]) # recreate the gW function
    print ' > Done.'
    print ' > Calculating error in autocorrelation'
    acn_diff = errors.acorrnErr(acn, w, n)                  # note acn not ac is used
    print ' > Done.'
    
    ac_label = r'$\bar{\bar{F}} = \langle\langle x^2\rangle_L\rangle_{HMC} = ' + r'{:.2f}; '.format(f_aav) \
        + r'\sigma_{\bar{\bar{F}}} = '+ r'{:.1f}'.format(f_diff) \
        + r'\sigma_{\sigma_{\bar{\bar{F}}}}' + r' = {:.1f}$'.format(f_ddiff)
    
    itau_label = r"$\tau_{\text{int}}(w_{\text{best}} = " + "{}) = ".format(int(w)) \
        + r"{:.2f} \pm {:.2f}$".format(itau, itau_diff)
    
    x = np.arange(0, itaus.size) # same size for itaus and acn
    lines = {0:(x[1:], g[:int(2*w)-1]), 1:(x, itaus, itaus_diff), 2:(x, acn, acn_diff)}
    labels = {1:itau_label}
    return lines, labels, w