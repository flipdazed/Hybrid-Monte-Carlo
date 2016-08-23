from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import random
from scipy import stats

from data import store
from common.utils import saveOrDisplay, prll_map

from correlations import acorr, errors
from models import Basic_KHMC as Model
from plotter import Pretty_Plotter, PLOT_LOC
import itertools

# generatte basic colours list
clist = [i for i in colors.ColorConverter.colors if i != 'w']
colour = [i for i in random.sample(clist, len(clist))]

# generate only dark colours
darkclist = [i for i in colors.cnames if 'dark' in i]
darkcolour = [i for i in random.sample(darkclist, len(darkclist))]
lightcolour = map(lambda strng: strng.replace('dark',''), darkcolour)

theory_colours = itertools.cycle(colour)
measured_colours = itertools.cycle(colour)

def plot(itau, pacc, acorr, op, tint_th=None, pacc_th=None, op_th=None, save=False):
    """Plots the two-point correlation function
    
    Required Inputs
        itau     :: {(x,y,e)} :: plots (x,y,e) as error bars
        pacc    :: {(x,y,e)} :: plots (x,y,e) as error bars
        # subtitle :: str  :: subtitle for the plot
        # op_name  :: str  :: the name of the operator for the title
        save     :: bool :: True saves the plot, False prints to the screen
    """
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    
    fig, ax = plt.subplots(3, sharex=True, figsize = (8, 8))
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(r"$\langle\rho_t\rangle_t$ and $\tau_{\text{int}}$ varying with $\delta\tau$ for $X^2$",
        fontsize=pp.ttfont)
    
    ax[0].set_title(r"HMC; lattice: $(100,)$; $n=20$; $m=0.1$; $M=10^5$; $\vartheta=\frac{\pi}{6}$", fontsize=pp.ttfont)
    
    ax[-1].set_xlabel(r'$\delta\tau$')
    
    ax[0].set_ylabel(r'$\tau_{\text{int}}$')
    x,y,e = itau
    ax[0].errorbar(x, y, yerr=e, ecolor='k', ms=3, fmt='o', alpha=0.5, label='Measured')
    if tint_th is not None:
        if len(tint_th) == 4:
            x,y,y_hi,y_lo = tint_th
            ax[0].fill_between(x, y_hi, y_lo, color='r', alpha=0.5, label='Theory')
        else:
            x,y = tint_th
        ax[0].plot(x, y, c='r', alpha=1, linestyle='--')
    
    ax[1].set_ylabel(r'$\langle\rho_t\rangle_t$')
    x,y,e = pacc
    ax[1].errorbar(x, y, yerr=e, ecolor='k', ms=3, fmt='o', alpha=0.5, label='Measured')
    if pacc_th is not None:
        x,y = pacc_th
        ax[1].plot(x, y, c='r', alpha=0.7, linestyle='--', label='Theory')

    ax[-1].set_ylabel(r'$\langle X^2_t\rangle_t$')
    x,y,e = op
    ax[-1].errorbar(x, y, yerr=e, ecolor='k', ms=3, fmt='o', alpha=0.5, label='Measured')
    if op_th is not None:
        x,y = op_th
        ax[-1].plot(x, y, c='r', alpha=0.7, linestyle='--', label='Theory')
    
    for a in ax: 
        a.legend(loc='best', shadow=True, fontsize = pp.axfont)
        xi,xf = a.get_xlim()
        a.set_xlim(xmin=xi-0.01*(xf-xi), xmax=xf+0.01*(xf-xi)) # give a decent view of the first point
        
        yi,yf = a.get_ylim()
        a.set_ylim(ymax=yf + .05*(yf-yi), ymin=yi-.05*(yf-yi)) # give 5% extra room at top
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in,
        mixing_angles, step_sizes, separations, opFn,
        tintTh=None, paccTh = None, opTh = None, plot_res=1000,
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
        max_sep     :: float :: define the max separation
    
    Optional Inputs
        rand_steps :: bool :: probability of with prob
        step_size :: float :: MDMC step size
        spacing :: float :: lattice spacing
        acFunc :: func :: function for evaluating autocorrelations
        save :: bool :: True saves the plot, False prints to the screen
    """
    
    # required declarations. If no lines are provided this still allows iteration for 0 times
    lines = {}  # contains the label as the key and a tuple as (x,y) data in the entry
    acs = {}
    
    rng = np.random.RandomState()
    
    print 'Running Model: {}'.format(file_name)
    
    def coreFunc(a):
        """runs the below for an angle, a
        Allows multiprocessing across a range of values for a
        """
        i,d,a = a
        model = Model(x0, pot=pot, rng=rng, step_size = d)
        
        c = acorr.Autocorrelations_1d(model)
        c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = a, verbose=False, verb_pos=i)
        acs = c.getAcorr(separations, opFn, norm = False)
        
        # get parameters generated
        pacc = c.model.p_acc
        pacc_err = np.std(c.model.sampler.accept.accept_rates)
        ans = errors.uWerr(c.op_samples, acorr=acs)         # get errors
        op_av, op_diff, _, itau, itau_diff, _, acns = ans             # extract data
        w = errors.getW(itau, itau_diff, n=n_samples)       # get window length
        if not np.isnan(w):
            acns_err = errors.acorrnErr(acns, w, n_samples)     # get error estimates
            acs = acns
        else:
            acns_err = np.full(acs.shape, np.nan)
            itau = itau_diff = np.nan
        
        return op_av, op_diff, acs, acns_err, itau, itau_diff, pacc, pacc_err, w
    
    # use multiprocessings
    elms = step_sizes.size*mixing_angles.size
    combinations = np.zeros((elms,3))
    combinations[:,1:] = np.array(np.meshgrid(step_sizes, mixing_angles)).T.reshape(-1,2)
    combinations[:,0] = np.arange(elms)
    
    ans = prll_map(coreFunc, combinations, verbose=True)
    
    print 'Finished Running Model: {}'.format(file_name)
    
    # results have now been obtained. This operation is a dimension shuffle
    # Currently the first iterable dimension contains one copy of each item
    # Want to split into separate arrays each of length n
    op_av, op_diff, acs, acns_err, itau, itau_diff, pacc, pacc_err, w = zip(*ans)
    
    op_av       = np.array(op_av)
    op_diff     = np.array(op_diff)
    acs         = np.array(acs)
    acns_err    = np.array(acns_err)
    itau        = np.array(itau)
    itau_diff   = np.array(itau_diff)
    pacc        = np.array(pacc)
    pacc_err    = np.array(pacc_err).ravel()
    w           = np.array(w)
    
    # Bundle all data ready for Plot() and store data as .pkl or .json for future use
    all_plot = {'itau':(step_sizes, itau, itau_diff), 
                'pacc':(step_sizes, pacc, pacc_err),
                'acorr':(step_sizes, acs, acns_err),
                'op':(step_sizes, op_av, op_diff)
                }
    
    max_step = step_sizes[-1]
    min_step = step_sizes[0]
    step_sizes_th = np.linspace(min_step, max_step, plot_res)
    if paccTh is not None:
        accs = np.asarray(map(paccTh,step_sizes_th))
        all_plot['pacc_th'] = (step_sizes_th, accs)
        
        if tintTh is not None:
            # itau = np.asarray([tintTh(dtau,acc) for dtau, acc in zip(step_sizes_th, pacc)])
            # itau_hi = np.asarray([tintTh(dtau,acc) for dtau, acc in zip(step_sizes_th, pacc+pacc_err)])
            # itau_lo = np.asarray([tintTh(dtau,acc) for dtau, acc in zip(step_sizes_th, pacc-pacc_err)])
            # all_plot['tint_th'] = (step_sizes_th, itau, itau_hi, itau_lo)
            itau = np.asarray([tintTh(dtau,acc) for dtau, acc in zip(step_sizes_th, accs)])
            all_plot['tint_th'] = (step_sizes_th, itau)
    # do the same for the X^2 operators
    if opTh is not None:
        all_plot['op_th'] = (step_sizes_th, np.asarray(map(opTh,step_sizes_th)))

    store.store(all_plot, file_name, '_allPlot')
    
    plot(save = saveOrDisplay(save, file_name), **all_plot)
    
if __name__ == '__main__':
    from hmc.potentials import Klein_Gordon as KG
    import numpy as np
    from theory.operators import x_sq, x2_1df
    from theory.acceptance import acceptance
    
    m         = 0.1
    
    file_name = __file__
    pot = KG(m=m)
    
    n, dim  = 20, 1
    x0 = np.random.random((n,)*dim)
    spacing = 1.
    
    # number of samples/burnin per point
    n_samples, n_burn_in = 100000, 1000
    
    mixing_angles = np.array([np.pi/8.])
    min_step = 0.1
    max_step = 0.7
    data_res = 100
    plot_res = 1000
    step_sizes = np.linspace(min_step, max_step, data_res)
    separations = np.arange(500)
    
    # theoretical functions
    tintTh  = None
    paccTn  = lambda dtau: acceptance(dtau=dtau, tau=dtau*1, n=n, m=m)
    opTh    = lambda null: x2_1df(m, n, spacing, 0)
    
    main(x0, pot, file_name, n_samples, n_burn_in, 
            mixing_angles, step_sizes, 
            separations = separations, opFn = x_sq,
            tintTh=tintTh, paccTh = paccTn, opTh = opTh,
            plot_res=plot_res, save = True)
    