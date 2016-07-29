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

# Fix colours/markers: A bug sometimes forces all colours the same
markers = []
for m in Line2D.filled_markers: # generate a list of markers
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass
# randomly order and create an iterable of markers to select from
marker = (i for i in random.sample(markers, len(markers)))

# do the same as above with colours but exclude lame colours
clist = [i for i in colors.ColorConverter.colors if i not in ['snow', 'white', 'k', 'w', 'r']]
colours = [i for i in random.sample(clist, len(clist))]

from scipy.optimize import curve_fit
def expFit(t, a, b, c):
    """Fit for an exponential curve
    Main parameter is t
    a,b,c are fitted w.r.t. f(t)
    """
    return a + b * np.exp(-t / c)

def plot(lines_d, x_lst, ws, subtitle, mcore, angle_labels, op_name, save):
    """Plots the two-point correlation function
    
    Required Inputs
        x_lst    :: list :: list of x_values for each angle that was run
        lines_d  :: {axis:[(y,label)]} :: plots (y,error,label) for each list item
        ws       :: list :: list of integration windows
        subtitle :: str  :: subtitle for the plot
        mcore    :: bool :: are there multicore operations? (>1 mixing angles)
        op_name  :: str  :: the name of the operator for the title
        angle_labels :: list :: the angle label text for legend plotting
        save     :: bool :: True saves the plot, False prints to the screen
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
        + r'$t = \langle\frac{\tau_{i+t} - \tau_{i}}{n}\rangle \frac{1}{\delta\tau}$')
    
    if not mcore: # don't want clutter in a multiple plot env.
        for a in range(1,len(ax)):  # Add the Window stop point as a red line
            # there is only one window item if not multiple lines
            ax[a].axvline(x=ws[0], linewidth=4, color='red', alpha=0.1)
    
    ax[0].set_ylabel(r'$g(w)$')
    ax[1].set_ylabel(r'$\tau_{\text{int}}(w)$')
    ax[2].set_ylabel(r'Autocorrelation, $\Gamma(t)$')
    
    #### plot for the 0th axis ####
    line_list = lines_d[0]
    axis = ax[0]
    colour = iter(colours)  # reset the iterable
    for x, lines in zip(x_lst, line_list):
        m = next(marker)        # get next marker style
        c = next(colour)        # get next colour
        
        # split into y function, errors in y and label
        y, e, l = lines # in this case there are no errors or labels used
        
        # allow plots in diff colours for +/
        yp = y.copy() ; yp[yp < 0] = np.nan     # hide negative values
        ym = y.copy() ; ym[ym >= 0] = np.nan    # hide positive values
        
        if not mcore:
            axis.scatter(x, yp, marker = 'o', color='g', linewidth=2., alpha=0.6, label=r'$g(t) \ge 0$')
            axis.scatter(x, ym, marker = 'x', color='r', linewidth=2., alpha=0.6, label=r'$g(t) < 0$')
        else:
            axis.plot(x, yp, color=c, lw=1., alpha=0.6)   # label with the angle
            axis.plot(x, ym, color='r', lw=1., alpha=0.6)
            once_label = None                     # set to blank so don't get multiple copies
    if not mcore: axis.legend(loc='best', shadow=True, fontsize = pp.axfont)
    
    #### plot the 1st axis ###
    # there is no angle label on the lines themselves on this axis
    # because the colours are synchronised across each plot
    # so the label on the bottom axis is enough
    line_list = lines_d[1]
    axis = ax[1]
    colour = iter(colours)      # reset the iterable
    for x, lines in zip(x_lst, line_list):
        m = next(marker)        # get next marker style
        c = next(colour)        # get next colour
        y, e, l, t = lines         # split into y function, errors, label and theory
        axis.fill_between(x, y-e, y+e, color=c, alpha=0.5)
        if t is not None:
            axis.axhline(y=t, linewidth=1, color = c, linestyle='--')
        # errorbar(x, y, yerr=e, markersize=3,
        #     color=c, fmt=m, alpha=0.5, ecolor='k')
        
        # Only add informative label if there is only one line
        # adds a pretty text box above the middle plot with info
        # contained in the variable l - assigned in preparePlot()
        if not mcore: pp.add_label(axis, l, fontsize=pp.tfont)
    
    #### plot the 2nd axis ###
    # This plot explicitly list the labels for all the angles
    line_list = lines_d[2]
    axis = ax[2]
    colour = iter(colours)      # reset the iterable
    for x, lines, a in zip(x_lst, line_list, angle_labels):
        m = next(marker)        # get next marker style
        c = next(colour)        # get next colour
        y, e, l = lines         # split into y function, errors in y and label
        try:    # errors when there are low number of sims
            axis.fill_between(x, y-e, y+e, color=c, alpha=0.5, label=a)
            # axis.errorbar(x, y, yerr=e, label = a,
#                 markersize=3, color=c, fmt=m, alpha=0.5, ecolor='k')
        except: # avoid crashing
            print 'Too few MCMC simulations to plot autocorrelations for: {}'.format(a)
    axis.legend(loc='best', shadow=True, fontsize = pp.axfont)
    
    #### start outdated section ####
    ## this won't work after the changes but shows the general idea of fitting a curve
    #
    # for i in range(1, len(lines)):              # add best fit lines
    #     x, y = lines[i][:2]
    #     popt, pcov = curve_fit(expFit, x, y)    # approx A+Bexp(-t/C)
    #     if not mcore:
    #         l_th = r'Fit: $f(t) = {:.1f} + {:.1f}'.format(popt[0], popt[1]) \
    #         + r'e^{-t/' +'{:.2f}'.format(popt[2]) + r'}$'
    #     else:
    #         l_th = None
    #     ax[i].plot(x, expFit(x, *popt), label = l_th,
    #         linestyle = '-', color=c, linewidth=2., alpha=.5)
    #### end outdated section ####
    
    # fix the limits so the plots have nice room 
    xi,xf = ax[2].get_xlim()
    ax[2].set_xlim(xmin= xi-.05*(xf-xi))    # decent view of the first point
    for a in ax:                            # 5% extra room at top & add legend
        yi,yf = a.get_ylim()
        a.set_ylim(ymax= yf+.05*(yf-yi), ymin= yi-.05*(yf-yi))
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, mixing_angle, angle_labels,
        opFn, op_name, rand_steps = False, step_size = .5, n_steps = 1,
        spacing = 1., itauFunc = None,
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
        angle_labels :: list :: list of angle label text for legend plotting
        opFn        :: func :: a function o run over samples
        op_name     :: str :: label for the operator for plotting
    
    Optional Inputs
        rand_steps :: bool :: probability of with prob
        step_size :: float :: MDMC step size
        n_steps :: int :: number of MDMC steps
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    rng = np.random.RandomState()
    multi_angle = len(mixing_angle) > 1         # see if multiprocessing is needed
    
    print 'Running Model: {}'.format(file_name)
    
    # output to print to screen
    out = lambda p,x,a:  '> measured at angle:{:3.1f}:'.format(a) \
        + ' <x^2>_L = {}; <P_acc>_HMC = {:4.2f}'.format(x, p)
    
    if not multi_angle:
        mixing_angle = mixing_angle[0]
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
        
        if itauFunc:
            t = itauFunc(tau=(n_steps*step_size), m=1, pa=p, theta=mixing_angle)
        else:
            t = None
        
        ans = errors.uWerr(cfn)                 # get the errors from uWerr
        x, gta, w = preparePlot(cfn, ans=ans, n=n_samples, theory=t, mcore=False)
        window_fns, int_ac_fns, acorr_fns = [[item] for item in gta]
        ws   = [w]                              # makes compatible with multiproc
        x_lst = [x]                             # again same as last two lines
        print out(p, xx, mixing_angle)
    
    else:   # use multicore support
        
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
            traj = c.model.traj         # get trajectory lengths for each LF step
            p = c.model.p_acc          # get acceptance rates at each M-H step
            xx = np.average(cfn)        # get average of the function run over the samples
            
            ans = errors.uWerr(cfn)
            if itauFunc:
                t = itauFunc(tau=n_steps*step_size, m=1, pa=p, theta=a)
            else:
                t = None
            print t
            x, gta, w = preparePlot(cfn, ans=ans, n = n_samples, theory=t, mcore = True)
            return xx, traj, p, x, gta, w
        #
        # use multiprocessing
        l = len(mixing_angle)                       # number of mixing angles
        ans = prll_map(coreFunc, zip(range(l), mixing_angle), verbose=False)
        
        # unpack from multiprocessing
        xx, traj, ps, x_lst, gtas, ws = zip(*ans)
        
        print '\n'*l                                # hack to avoid text overlapping in terminal
        for p, x, a in zip(ps, xx, mixing_angle):   # print intermediate results to screen
            print out(p,x,a)
        window_fns, int_ac_fns, acorr_fns = zip(*gtas)  # separate out to respective lists
    
    lines = {0:window_fns, 1:int_ac_fns, 2:acorr_fns}   # create a dictionary for plotting
    #
    print 'Finished Running Model: {}'.format(file_name)
    
    subtitle = r"Potential: {}; Lattice Shape: ".format(pot.name) \
        + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
            x0.shape, spacing, step_size, n_steps)
    
    # all_plot contains all necessary keyword arguments
    all_plot = {'lines_d':lines,'x_lst':x_lst, 'ws':ws, 'subtitle':subtitle, 'mcore':multi_angle,
        'angle_labels':angle_labels, 'op_name':op_name}
    
    store.store(all_plot, file_name, '_allPlot')
    
    plot(lines_d=lines,x_lst=x_lst,ws=ws,subtitle=subtitle,mcore=multi_angle,
        angle_labels=angle_labels,op_name=op_name,
        save = saveOrDisplay(save, file_name)
        )
    pass
#
def preparePlot(op_samples, ans, n, theory=None, mcore=False):
    """Prepares the plot according to the output of uWerr
    
    Required Inputs
        op_samples :: np.ndarray :: function acted upon the HMC samples
        ans :: tuple :: output form uWerr
        n   :: int   :: number of samples from MCMC
        theory   :: float :: theoretical iTau value
    
    Optional Input
        mcore :: bool :: flag that ans is a nested list of l_ans = [ans, ans, ...]
    """
    
    f_aav, f_diff, f_ddiff, itau, itau_diff, itaus, acn = ans
    
    w = round((f_ddiff/f_diff)**2*n - .5, 0)            # obtain the best windowing point
    l = min(itaus.size, int(2*w)+1)
    fn = lambda t: acorr.acorr(op_samples=op_samples, mean=f_aav, separation=t)
    ac  = np.asarray([fn(t=t) for t in range(0, l)])    # calculate autocorrelations
    itaus_diff  = errors.itauErrors(itaus, n=n)         # calcualte error in itau
    
    g_int = np.cumsum(ac[1:l]/ac[0])                    # recreate the g_int function
    g = np.asarray([errors.gW(t, v, 1.5, n) for t,v in enumerate(g_int,1)]) # recreate the gW function
    acn_diff = errors.acorrnErr(acn, w, n)                  # note acn not ac is used
    
    itau_label = r"$\tau_{\text{int}}(w_{\text{best}} = " + r"{}) = ".format(int(w)) \
        + r"{:.2f} \pm {:.2f}$".format(itau, itau_diff)
    x = np.arange(l) # same size for itaus and acn
    
    g = np.insert(g,0,np.nan)   # g starts at x[1] not x[0]
    
    # create items for plotting
    gta = (g, None, ''), (itaus[:l], itaus_diff[:l], itau_label, theory), (acn[:l], acn_diff[:l], '')
    return x, gta, w