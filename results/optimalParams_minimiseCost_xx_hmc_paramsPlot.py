from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from results.data import store
from plotter import Pretty_Plotter, PLOT_LOC, mlist
import itertools
import random

# generatte basic colours list
clist = [i for i in colors.ColorConverter.colors if i != 'w']
colour = [i for i in random.sample(clist, len(clist))]

# generate only dark colours
darkclist = [i for i in colors.cnames if 'dark' in i]
darkcolour = [i for i in random.sample(darkclist, len(darkclist))]
lightcolour = map(lambda strng: strng.replace('dark',''), darkcolour)
# randomly order and create an iterable of markers to select from
marker = [i for i in random.sample(mlist, len(mlist))]

theory_colours = itertools.cycle(colour)
measured_colours = itertools.cycle(colour)
markers = itertools.cycle(marker)

def plot(lines, save=False):
    """Plots the two-point correlation function
    
    Required Inputs
        itau     :: {(x,y,e)} :: plots (x,y,e) as error bars
        pacc    :: {(x,y,e)} :: plots (x,y,e) as error bars
        # subtitle :: str  :: subtitle for the plot
        # op_name  :: str  :: the name of the operator for the title
        save     :: bool :: True saves the plot, False prints to the screen
    """
    
    pp = Pretty_Plotter()
    pp.s = 1.5
    pp._teXify() # LaTeX
    
    fig, ax = plt.subplots(1, figsize = (10, 8))
    ax = [ax]
    fig.suptitle(r"Showing that $n$ has little effect on $\tau_{\text{int}}$",
        fontsize=pp.ttfont+2)
    
    ax[0].set_title(r"HMC; lattice: $(100,)$; $m=0.01$; $M=10^5$; $\vartheta=\frac{\pi}{2}$", fontsize=pp.ttfont)
    
    ax[-1].set_xlabel(r'$\delta\tau$')
    
    ax[0].set_ylabel(r'$\tau_{\text{int}}$')
    for line in lines:
        m = next(markers)
        c = next(measured_colours)
        
        label = line['n_steps']
        x = line['step_size']
        y = line['itau']
        ax[0].scatter(x, y, alpha=0.5, c=c, marker=m, label=int(label))
    
    for a in ax:
        a.legend(loc='best', shadow=True, fontsize = pp.axfont)
        xi,xf = a.get_xlim()
        a.set_xlim(xmin=xi-0.01*(xf-xi), xmax=xf+0.01*(xf-xi)) # give a decent view of the first point
        
        yi,yf = a.get_ylim()
        a.set_ylim(ymax=yf + .05*(yf-yi), ymin=yi-.05*(yf-yi)) # give 5% extra room at top
    
    ax[0].legend(bbox_to_anchor=(0., -0.3, 1., .102), loc=9,
           ncol=6, mode="expand", borderaxespad=0.)
    fig.subplots_adjust(bottom=0.3)
    
    pp.save_or_show(save, PLOT_LOC)
    pass

if __name__ == '__main__':
    from results.common.utils import saveOrDisplay
    
    file_name = __file__
    save = True
    
    # load the existing parameters
    dest = "results/optimalParams_minimiseCost_xx_hmc_params_coarse.pkl"
    output = store.load(dest)
    
    # in this case best_params: [best_theta, best_]
    best_params, best_fn, arg_grid, fn_grid = output
    
    print '\n\n__Model details__'
    print 'Potential:\tFree-Field'
    print 'Lattice:\t(100,1)'
    print 'Mass:\t\t0.01'
    print '\nBest parameters:\n\tstep_size: {:10.5f}\n\tn_steps:   {:10.5f}\n\ttau:       {:10.5f}'.format(
            best_params[0], best_params[1], np.prod(best_params))
    
    x,y = arg_grid
    z   = fn_grid
    l = z.shape[0] # this is the grid length (square grid)
    
    # create an n by 3 array shape == (n,3)
    n_by_3 = np.column_stack((y.ravel(), x.ravel(), z.ravel()))
    
    lines = []
    for i in range(l):
        line = {}
        line['n_steps']    = n_by_3[i][0]      # first index is repeated
        line['step_size']  = n_by_3[::l][:,1]  # second index is uniqe every l entries
        line['itau']       = n_by_3[::l][:,2]  # same with third
        lines.append(line)
    
    plot(save=saveOrDisplay(save, file_name), lines=lines)