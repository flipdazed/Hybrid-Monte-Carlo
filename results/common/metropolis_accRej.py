# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt

from plotter import Pretty_Plotter, PLOT_LOC
from models import Basic_GHMC as Model
from utils import saveOrDisplay

def plot(probs, accepts, h_olds, h_news, exp_delta_hs, 
    subtitle, labels, save,
    h_mdmc=None, mdmc_deltaH=None, # for mdmc additions
    ):
    """Plots 3 stacked figures:
        1. the acceptance probability at each step
        2. the hamiltonian (old, new) at each step
        3. the exp{-delta H} at each step
    Overlayed with red bars is each instance in which a configuration was rejected
    by the Metropolis-Hastings accept/reject step
    
    Required Inputs
        probs           :: np.array :: acc. probs
        accepts         :: np.array :: array of boolean acceptances (True = accepted)
        h_olds          :: np.array :: old hamiltonian at each step
        h_news          :: np.array :: new hamiltonian at each step
        exp_delta_hs    :: np.array :: exp{-delta H} at each step
        h_mdmc          :: np.array :: 
        mdmc_deltaH     :: np.array ::
        subtitle        :: str      :: the subtitle to put in ax[0].set_title()
        save            :: bool :: True saves the plot, False prints to the screen
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig, ax = plt.subplots(3, sharex=True, figsize = (8, 8))
    fig.suptitle(r'Data from {} Metropolis acceptance steps'.format(len(probs)), 
        fontsize=pp.ttfont)
    
    fig.subplots_adjust(hspace=0.2)
    
    l = probs.size          # length of HMC trajectory - lots rely on this being at the top
    
    ### Add top pseudo-title and bottom shared x-axis label
    ax[0].set_title(subtitle, fontsize=pp.tfont)
    ax[-1].set_xlabel(r'HMC trajectory, $t$')
    
    ### add the rejection points in the background
    xrng = range(1, l+1)    # calculate xrange to match length
    for a in ax: # iterate over each axis
        for x,val in zip(xrng,accepts): # iterate over all rejection points
            if val == False: a.axvline(x=x, linewidth=4, color='red', alpha=0.2)
    
    ### add the lines to the plots
    ax[0].set_ylabel(r'Hamiltonian, $H$')
    ax[0].plot(xrng, h_olds, linestyle='-', color='blue', linewidth=2., alpha=0.4, 
        label=r'$H(t-1)$')
    ax[0].plot(xrng, h_news, linestyle='-', color='green', linewidth=2., alpha=0.4, 
        label=r'$H(t)$')
    
    ax[1].set_ylabel(r'$\exp{-\delta H}$')
    
    ax[1].plot(xrng, exp_delta_hs, linestyle='-', color='blue', linewidth=2., alpha=0.4,
        label=r'$\exp{ -\delta H(t)}$')
    
    ax[2].set_ylabel(r'Acceptance Prob., $P_{\text{acc}}$')
    ax[2].plot(xrng, probs, linestyle='-', color='blue', linewidth=2., alpha=0.6)
    
    ### test to see if we will plot all the intermediate MDMC steps
    plot_mdmc = (h_mdmc is not None) & (mdmc_deltaH is not None)
    
    if plot_mdmc: # all these functions rely on l being calculated!!
        # functions to calculate the staggering of y and x
        # for the MDMC. Staggering means calculating the start and
        # end points of each single trajectory consisting of 1 MDMC integration
        n = h_mdmc.size/l - 1 # this calculates n_steps for the MDMC (in a backwards way)   
        staggered_xi = np.arange(0, l)
        staggered_xf = np.arange(1, l+1)
        staggered_y = lambda arr, offset: arr[0+offset::(n+1)]
        remove_links = lambda arr: [None if (i)%(n+1) == 0 else a for i,a in enumerate(arr)]
    
        ## calculate a linear space as a fraction of the HMC trajectory
        mdmc_x = np.asarray([np.linspace(i, j, n+1) for i,j in zip(range(0,l), range(1,l+1))])
        mdmc_x = mdmc_x.flatten()
        
        ax[0].plot(mdmc_x, remove_links(h_mdmc), linestyle='--', color='green', linewidth=1, alpha=1, 
            label=r'MDMC: $H(t+\epsilon)$')
        ax[0].scatter(staggered_xi, staggered_y(h_mdmc,0), color='red', marker='o', alpha=1, 
            label=r'Start MDMC')
        ax[0].scatter(staggered_xf, staggered_y(h_mdmc,n), color='red', marker='x', alpha=1, 
            label=r'End MDMC')
        
        ax[1].plot(mdmc_x, remove_links(mdmc_deltaH), linestyle='--', color='blue', linewidth=1., alpha=1,
            label=r'MDMC: $\exp{ -\delta H(t+\epsilon)}$')
        ax[1].scatter(staggered_xi, staggered_y(mdmc_deltaH, 0), color='red', marker='o', 
            alpha=1, label=r'Start MDMC')
        ax[1].scatter(staggered_xf, staggered_y(mdmc_deltaH, n), color='red', marker='x',
            alpha=1, label=r'End MDMC')
    
    ### adds labels to the plots
    for i, text in labels.iteritems(): pp.add_label(ax[i], text, fontsize=pp.tfont)
    
    ### place legends
    ax[0].legend(loc='upper left', shadow=True, fontsize = pp.ipfont, fancybox=True)
    ax[1].legend(loc='upper left', shadow=True, fontsize = pp.ipfont, fancybox=True)
    
    ### formatting
    for i in ax: i.grid(True) 
    # ax[1].set_yscale("log", nonposy='clip') # set logarithmic y-scale
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, accept_kwargs, save = False, 
        step_size=0.05, n_steps=20, spacing=1., mixing_angle=np.pi/2., 
        accept_all = False, plot_mdmc=False):
    """A wrapper function
    
    Required Inputs
        x0              :: np.array :: initial position input to the HMC algorithm
        pot             :: pot. cls :: defined in hmc.potentials
        file_name       :: string   :: the final plot will be saved with a similar name if save=True
        n_samples       :: int      :: number of HMC samples
        n_burn_in       :: int      :: number of burn in samples
        accept_kwargs   :: dict     :: required for Model() see docs in dynamics
    
    Optional Inputs
        save :: bool    :: bool     :: True saves the plot, False prints to the screen
        step_size       :: int      :: Leap Frog step size
        n_steps         :: int      :: LF trajectory lengths
        spacing         :: float    :: lattice spacing
        mixing_angle    :: float    :: np.pi/2 is HMC, 0 is no mixing, other configs are GHMC
        accept_all      :: bool     :: accept all proposed p,x configs i.e. bypass M-H step
        plot_mdmc       :: bool     :: plots all intermediate MDMC steps
    """
    
    model = Model(x0, pot, step_size=step_size, n_steps=n_steps, spacing=spacing, accept_kwargs=accept_kwargs)
    
    # plots out mdmc path in full for each HMC trajctory
    model.sampler.dynamics.save_path = plot_mdmc
    
    print 'Running Model: {}'.format(file_name)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True, 
        mixing_angle=mixing_angle)
    print 'Finished Running Model: {}'.format(file_name)
    
    #### This is pulling data from the Metropolis-Hastings Steps
    # pull out all the data from the metropolis class
    accept_rates = np.asarray(model.sampler.accept.accept_rates[n_burn_in:]).flatten()
    accept_rejects = np.asarray(model.sampler.accept.accept_rejects[n_burn_in:]).flatten()
    delta_hs = np.asarray(model.sampler.accept.delta_hs[n_burn_in:]).flatten()
    h_olds = np.asarray(model.sampler.accept.h_olds[n_burn_in:]).flatten()
    h_news = np.asarray(model.sampler.accept.h_news[n_burn_in:]).flatten()
    
    # get the means where relevent
    av_acc      = np.asscalar(accept_rates.mean())
    meas_av_acc = np.asscalar(accept_rejects.mean())
    meas_av_exp_dh  = np.asscalar(np.exp(-delta_hs).mean())
    exp_delta_hs = np.exp(-delta_hs)
    print '\n\t<Prob. Accept>: {:4.2f}'.format(av_acc)
    print '\t<Prob. Accept>: {:4.2f}     (Measured)'.format(meas_av_acc)
    print '\t<exp{{-𝛿H}}>:     {:8.2E} (Measured)\n'.format(meas_av_exp_dh)
    print '\tstep size:      {:4.2f}'.format(model.sampler.dynamics.step_size)
    print '\ttraj. length:   {:4.2f}'.format(model.sampler.dynamics.n_steps)
    
    pow_ten = int(np.floor(np.log10(meas_av_exp_dh)))   # calculate N for AeN sci. format
    decimal = meas_av_exp_dh / float(10**pow_ten)       # calculate A for AeN
    
    # one long subtitle - long as can't mix LaTeX and .format()
    subtitle = '\centering Potential: {}, Lattice: {}'.format(pot.name, x0.shape) \
        + ', Accept: {}'.format(['M-H','All'][accept_all]) \
        + r', $\theta = ' + '{:3.1f}$'.format(mixing_angle) \
        + r', $\epsilon = ' + '{:4.2f}$'.format(step_size) \
        + r', $n = ' + '{}$'.format(n_steps)
    
    # text for pretty labels in each subplot
    ax2_label = r'$\langle P_{\text{acc}} \rangle='+' {:4.2f}$'.format(av_acc)
    ax1_label = r' $\langle \exp{ -\delta H}  \rangle='+' {:3.1f}'.format(decimal) \
        + r'\times 10^{' + '{}'.format(pow_ten) + '}$'
    
    ### This is pulling data from the MDMC steps
    if plot_mdmc:
        p_mdmc = model.sampler.dynamics.p_ar # contains start step from each previous
        x_mdmc = model.sampler.dynamics.x_ar # so shape is n_steps + 1
    
        # these are both lists so need to vonvert back to ararys
        kE_mdmc = map(model.sampler.potential.kE, p_mdmc) # calculate kE
        uE_mdmc = map(model.sampler.potential.uE, x_mdmc) # calculate uE
        h_mdmc = np.asarray(np.asarray(kE_mdmc) + np.asarray(uE_mdmc))
        
        # filters out all the values multiples of n_steps
        h_mdmc_0 = np.repeat(h_mdmc[0::(n_steps+1)], n_steps+1)
        delta_H = h_mdmc - h_mdmc_0
        mdmc_deltaH = np.exp(-delta_H)
        
    else:
        h_mdmc = mdmc_deltaH = None
    
    plot(accept_rates, accept_rejects, h_olds, h_news, exp_delta_hs,
        subtitle = subtitle,
        save = saveOrDisplay(save, file_name),
        labels = {1:ax1_label, 2:ax2_label},
        # include mdmc if plot_mdmc
        h_mdmc = h_mdmc, mdmc_deltaH = mdmc_deltaH
        )
    pass