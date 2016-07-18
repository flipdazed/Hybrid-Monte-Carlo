# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

from data import store
from correlations import acorr, corr, errors
from models import Basic_GHMC as Model
from utils import saveOrDisplay, prll_map
from plotter import Pretty_Plotter, PLOT_LOC

from matplotlib.lines import Line2D

def plot(lines, w, subtitle, mcore, angle_labels, labels, op_name, save):
    """Plots the two-point correlation function
    
    Required Inputs
        acs              :: {axis:(x,y)} :: plots (x,y,error)
        lines            :: {axis:(x,y)} :: plots (x,y,error)
        subtitle :: str  :: subtitle for the plot
        mcore    :: bool :: are there multicore operations? (>1 mixing angles)
        angle_labels :: list :: list of angle label text for legend plotting
        labels   :: {axis:label} :: a dictionary to add labels to a specific axis
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
    
    # Add top pseudo-title and bottom shared x-axis label
    ax[0].set_title(subtitle, fontsize=pp.tfont)
    ax[-1].set_xlabel(r'Av. trajectories between samples, '\
        + r'$t = \langle \tau_{i+t} - \tau_{i} \rangle / n\delta\tau$')
    
    if not mcore: # then can iterate safely
        lines_list = [lines]
        w_list = [w]
        angle_labels = [angle_labels]
        labels_list = [labels]
        # don't want clutter in a multiple plot env.
        for a in range(1,len(ax)):  # Add the Window stop point as a red line
            ax[a].axvline(x=w, linewidth=4, color='red', alpha=0.1)
        
    else: # already acked for iteration if multicored up
        lines_list = lines
        w_list = w
        angle_labels = angle_labels
        labels_list = labels
    ax[0].set_ylabel(r'$g(w)$')
    ax[1].set_ylabel(r'$\tau_{\text{int}}(w)$')
    ax[2].set_ylabel(r'Autocorrelation, $\Gamma(t)$')
    
    marker = (i for i in random.sample(markers, len(markers)))
    # Fix colours: A bug sometimes forces all colours the same
    clist = [i for i in colors.ColorConverter.colors if i not in ['snow', 'white', 'k', 'w', 'r']]
    colour = (i for i in random.sample(clist, len(clist)))
    
    j = 1   # show every j points
    k = None # premature end
    
    for i, (w, lines, a, labels) in enumerate(zip(w_list, lines_list, angle_labels, labels_list)):
        c = next(colour)
        m = next(marker)
        x, y = lines[0]                         # allow plots in diff colours for +/-
        yp = y.copy() ; yp[yp < 0] = np.nan     # hide negative
        ym = y.copy() ; ym[ym >= 0] = np.nan    # hide positive
        if not mcore:
            ax[0].scatter(x[:k], yp[:k], marker = 'o', color='g', linewidth=2., alpha=0.6, label=r'$g(t) \ge 0$')
            ax[0].scatter(x[:k], ym[:k], marker = 'x', color='r', linewidth=2., alpha=0.6, label=r'$g(t) < 0$')
        else:
            one_off_label = r'$g(t,\theta) < 0$' if i == len(w_list) else None
            ax[0].plot(x[:k], yp[:k], color=c, linewidth=1., alpha=0.6, label = a)
            ax[0].plot(x[:k], ym[:k], color='r', linewidth=1., alpha=0.6, label = one_off_label)
        
        x, y, e = lines[1]
        if not mcore:
            l = r'MCMC data; '
            tint_label = r''
        else:
            l = r''
            tint_label = r': ' + labels[1] # this should always be 1 entry in the dict but bad programming!
        ax[1].errorbar(x[:k:j], y[:k:j], yerr=e[:k:j], label = l + a + tint_label,
            markersize=3, color=c, fmt=m, alpha=0.5, ecolor='k')
        
        x, y, e = lines[2]
        try:    # errors when there are low number of sims
            ax[2].errorbar(x[:k:j], y[:k:j], yerr=e[:k:j], label = l + a,
                markersize=3, color=c, fmt=m, alpha=0.5, ecolor='k')
        except: # avoid crashing
            print 'Too few MCMC simulations to plot autocorrelations for: {}'.format(a)
        
        if not mcore:
            
            # adds labels to the plots
            for i, text in labels.iteritems(): pp.add_label(ax[i], text, fontsize=pp.tfont)
        
        for i in range(1, len(lines)):              # add best fit lines
            x, y = lines[i][:2]
            popt, pcov = curve_fit(expFit, x, y)    # approx A+Bexp(-t/C)
            if not mcore: 
                l_th = r'Fit: $f(t) = {:.1f} + {:.1f}'.format(popt[0], popt[1]) \
                + r'e^{-t/' +'{:.2f}'.format(popt[2]) + r'}$'
            else:
                l_th = None
            ax[i].plot(x[:k], expFit(x[:k], *popt), label = l_th,
                linestyle = '-', color=c, linewidth=2., alpha=.5)
    
    # fix the limits so the plots have nice room 
    xi,xf = ax[2].get_xlim()
    ax[2].set_xlim(xmin= xi-.05*(xf-xi))    # decent view of the first point
    for a in ax:                            # 5% extra room at top & add legend
        yi,yf = a.get_ylim()
        a.set_ylim(ymax= yf+.05*(yf-yi), ymin= yi-.05*(yf-yi))
        a.legend(loc='best', shadow=True, fontsize = pp.axfont)
    
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
        angle_fracs :: iterable :: list of values: n, for mixing angles of π/n
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
    rng = np.random.RandomState()
    angls = np.pi/angle_fracs
    
    print 'Running Model: {}'.format(file_name)
    
    def coreFunc(a):
        """runs the below for an angle, a"""
        i,a = a
        model = Model(x0, pot=pot, spacing=spacing, # set up model
            rng=rng, step_size = step_size,
            n_steps = n_steps, rand_steps=rand_steps)
        c = acorr.Autocorrelations_1d(model)                    # set up autocorrs
        c.runModel(n_samples=n_samples, n_burn_in=n_burn_in,    # run MCMC
            mixing_angle = a, verbose=True, verb_pos=i)
        cfn = opFn(c.model.samples)                             # run func on HMC samples
        
        # get parameters generated
        p = c.model.p_acc           # get acceptance rates at each M-H step
        xx = np.average(cfn)        # get average of the function run over the samples
        return cfn, xx, p
        
    ans = prll_map(coreFunc, zip(range(angls.size), angls), verbose=False)
    cfn_lst, xx_lst, p_lst = zip(*ans)          # unpack from multiprocessing
    print '\n'*angls.size                       # hack to avoid overlapping!
    
    store.store(cfn_lst, file_name, '_cfn')     # store the function
    store.store(angls, file_name, '_angles')    # store angles used
    
    print '\n > Calculating integrated autocorrelations...\n'
    def coreFunc2(cfn):
        ans = errors.uWerr(cfn)
        _, f_diff, _, itau, itau_diff, itaus, _ = ans
        w = errors.getW(itau, itau_diff, n=cfn.size)
        out = (itau, itau_diff, f_diff, w)
        return out
    
    ans = prll_map(coreFunc2, cfn_lst, verbose=True)
    itau_lst, itau_diffs_lst, f_diff_lst, w_lst = zip(*ans) # unpack from multiprocessing
    
    print 'Finished Running Model: {}'.format(file_name)
    
    subtitle = r"Potential: {}; Lattice Shape: ".format(pot.name) \
        + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
            x0.shape, spacing, step_size, r'$1$ (KHMC)')
    
    lines = { # format is (y, Errorbar) if no errorbars then None
            0:(itau_lst, itau_diffs_lst), 
            3:(xx_lst, f_diff_lst),
            1:(w_lst, None),
            2:(p_lst, None)
            }
    
    all_plot = {'lines':lines, 'x_vals':angls, 'subtitle':subtitle, 'op_name':op_name}
    store.store(all_plot, file_name, '_allPlot')
    
    # enter as keyword arguments
    # plot(**all_plot, save = saveOrDisplay(save, file_name))
    pass