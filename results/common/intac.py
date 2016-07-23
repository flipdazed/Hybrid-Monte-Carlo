# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import random

from data import store
from correlations import acorr, corr, errors
from models import Basic_GHMC as Model
from utils import saveOrDisplay, prll_map
from plotter import Pretty_Plotter, PLOT_LOC
from matplotlib.collections import LineCollection

ignorecols = ['snow', 'white', 'k', 'w', 'r']
clist = [i for i in colors.ColorConverter.colors if i not in ignorecols]

def plot(x, lines, subtitle, op_name, save):
    """Plots the two-point correlation function
    
    Required Inputs
        x_vals   :: list :: list / np.ndarray of angles the routine was run for
        lines            :: {axis:(y, error)} :: contains (y,error) for each axis
        subtitle :: str  :: subtitle for the plot
        op_name  :: str  :: the name of the operator for the title
        save     :: bool :: True saves the plot, False prints to the screen
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = r"\usepackage{amsmath}"
    pp._updateRC()
    ax =[]
    
    fig, ax = plt.subplots(4, sharex=True, figsize = (8, 8))
    fig.suptitle(r"Integrated Autocorrelations for " \
        + r"{} and varying $\theta$".format(op_name), fontsize=pp.ttfont)
    
    fig.subplots_adjust(hspace=0.2)
    
    # Add top pseudo-title and bottom shared x-axis label
    ax[0].set_title(subtitle, fontsize=pp.tfont)
    ax[-1].set_xlabel(r'Mixing angle, $\theta$')
    ax[0].set_ylabel(r'$\bar{\bar{\tau}}_{\text{int}}$')
    ax[1].set_ylabel(r'$\langle \hat{O} \rangle_{L,\tau}$')
    ax[2].set_ylabel(r'Int. window, $w$')
    ax[3].set_ylabel(r'$\langle P_{\text{acc}}\rangle$')
    
    def formatFunc(tic, tic_loc):
        return r"${}\pi$".format(tic)
    ax[-1].get_xaxis().set_major_formatter(ticker.FuncFormatter(formatFunc))
    # Fix colours: A bug sometimes forces all colours the same
    colour = (i for i in random.sample(clist, len(clist))) # defined top of file
    for k, v in lines.iteritems():
        for (y, err, label) in v:
            c = next(colour)
            if err is None:
                ax[k].plot(x, y, c=c, lw=1., alpha=0.6, label=label)
            else:
                y = np.asarray(y); err = np.asarray(err)
                ax[k].fill_between(x, y-err, y+err, color=c, alpha=0.9)
                # ax[k].errorbar(x, y, yerr=err, c=c, ecolor='k', ms=3, fmt='o', alpha=0.5,
                #     label=label)
    # Fix the limits so the plots have nice room 
    for a in ax:                            # 5% extra room at top & add legend
        # xi,xf = a.get_xlim()
        # a.set_xlim(xmin= xi-.05*(xf-xi))    # decent view of the first point
        # a.set_xlim(-0.001, 0.101)
        yi,yf = a.get_ylim()
        a.set_ylim(ymax= yf+.05*(yf-yi), ymin= yi-.05*(yf-yi))
        # a.legend(loc='best', shadow=True, fontsize = pp.axfont)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, angle_fracs,
        opFn, op_name, rand_steps = True, step_size = .1, n_steps = 1, spacing = 1., 
        save = False):
    """Takes a function: opFn. Runs HMC-MCMC. Runs opFn on GHMC samples.
        Calculates Integrated Autocorrelation + Errors across a number of angles
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        angle_fracs :: iterable :: list of values: n, for mixing angles of nπ
        opFn        :: func :: a function o run over samples
        op_name     :: str :: label for the operator for plotting
    
    Optional Inputs
        rand_steps  :: bool :: probability of with prob
        step_size   :: float :: MDMC step size
        n_steps     :: int :: number of MDMC steps
        spacing     :: float :: lattice spacing
        save        :: bool :: True saves the plot, False prints to the screen
    """
    if not isinstance(angle_fracs, np.ndarray): angle_fracs = np.asarray(angle_fracs)
    if not hasattr(n_samples, '__iter__'): n_samples = [n_samples]*angle_fracs.size
    rng = np.random.RandomState()
    angls = np.pi*angle_fracs
    
    print 'Running Model: {}'.format(file_name)
    explicit_prog = (angls.size <= 16)
    def coreFunc(a):
        """runs the below for an angle, a"""
        i,a, n_samples = a
        model = Model(x0, pot=pot, spacing=spacing, # set up model
            rng=rng, step_size = step_size,
            n_steps = n_steps, rand_steps=rand_steps)
        c = acorr.Autocorrelations_1d(model)                    # set up autocorrs
        c.runModel(n_samples=n_samples, n_burn_in=n_burn_in,    # run MCMC
            mixing_angle = a, verbose=explicit_prog, verb_pos=i)
        cfn = opFn(c.model.samples)                             # run func on HMC samples
        
        # get parameters generated
        p = c.model.p_acc           # get acceptance rates at each M-H step
        
        # Calculating integrated autocorrelations
        ans = errors.uWerr(cfn)
        xx, f_diff, _, itau, itau_diff, itaus, _ = ans
        w = errors.getW(itau, itau_diff, n=cfn.shape[0])
        out = itau, itau_diff, f_diff, w
        return xx, p, itau, itau_diff, f_diff, w
    
    ans = prll_map(coreFunc, zip(range(angle_fracs.size), angls, n_samples), verbose=1-explicit_prog)
    
    # unpack from multiprocessing
    xx_lst, p_lst, itau_lst, itau_diffs_lst, f_diff_lst, w_lst = zip(*ans)
    print '\n'*angle_fracs.size*explicit_prog         # hack to avoid overlapping!
    
    print 'Finished Running Model: {}'.format(file_name)
    
    subtitle = r"Potential: {}; Lattice: ".format(pot.name) \
        + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
            x0.shape, spacing, step_size, n_steps)
    
    lines = { # format is [(y, Errorbar, label)] if no errorbars then None
            0:[(itau_lst, itau_diffs_lst, r'Measured')], 
            1:[(xx_lst, f_diff_lst, r'Measured')],
            2:[(w_lst, None, r'Measured')],
            3:[(p_lst, None, r'Measured')]
            }
    
    all_plot = {'lines':lines, 'x':angle_fracs, 'subtitle':subtitle, 'op_name':op_name}
    store.store(all_plot, file_name, '_allPlot')
    
    # enter as keyword arguments
    plot(save = saveOrDisplay(save, file_name), **all_plot)
    pass