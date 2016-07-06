# -*- coding: utf-8 -*- 
import numpy as np
import os
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from models import Basic_GHMC as Model
from utils import saveOrDisplay

def plot(probs, accepts, h_olds, h_news, exp_delta_hs, h_mdmc, mdmc_deltaH, 
    subtitle, save):
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
        subtitle        :: str      :: the subtitle to put in ax[0].set_title()
        save            :: bool :: True saves the plot, False prints to the screen
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig, ax = plt.subplots(3, sharex=True, figsize = (8, 8))
    fig.suptitle(r'Data from {} Metropolis acceptance steps'.format(len(probs)), fontsize=16)
    
    fig.subplots_adjust(hspace=0.1)
    l = probs.size
    n = h_mdmc.size/l - 1
    mdmc_x = np.asarray([np.linspace(i, j, n+1) for i,j in zip(range(0,l), range(1,l+1))])
    mdmc_x = mdmc_x.flatten()
    xrng = range(1, l+1)
    for a in ax:
        for x,val in zip(xrng,accepts):
            if val == False: a.axvline(x=x, linewidth=4, color='red', alpha=0.2)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    ax[0].set_ylabel(r'Acceptance Prob., $P_{\text{acc}}$')
    ax[0].plot(xrng, probs, linestyle='-', color='blue', linewidth=2., alpha=0.6)
    ax[0].grid(True)
    
    ax[1].set_ylabel(r'Hamiltonian, $H$')
    ax[1].plot(xrng, h_olds, linestyle='-', color='blue', linewidth=2., alpha=0.4, 
        label=r'$H_{\text{i-1}}$ M-H')
    ax[1].plot(xrng, h_news, linestyle='-', color='green', linewidth=2., alpha=0.4, 
        label=r'$H_{\text{i}}$ M-H')
    ax[1].plot(mdmc_x, h_mdmc, linestyle='--', color='green', linewidth=1, alpha=1, 
        label=r'$H_{\delta i}$ MDMC')
    
    staggered_xi = np.arange(0, l)
    staggered_xf = np.arange(1, l+1)
    staggered_y = lambda arr, offset: arr[0+offset::(n+1)]
    
    ax[1].scatter(staggered_xi, staggered_y(h_mdmc,0), color='red', marker='o', alpha=1, 
        label=r'Start MDMC')
    ax[1].scatter(staggered_xf, staggered_y(h_mdmc,n), color='red', marker='x', alpha=1, 
        label=r'End MDMC')
    ax[1].legend(loc='best', shadow=True, fontsize = pp.axfont-3)
    ax[1].grid(True)
    
    ax[2].set_ylabel(r'$\exp{-\delta H}$')
    ax[2].plot(xrng, exp_delta_hs, linestyle='-', color='blue', linewidth=2., alpha=0.4,
        label=r'$\exp{ -\delta H_{\text{HMC }}}$')
    
    ax[2].plot(mdmc_x, mdmc_deltaH, linestyle='--', color='blue', linewidth=1., alpha=1,
        label=r'$\exp{ -\delta H_{\text{MDMC}}}$')
    ax[2].scatter(staggered_xi, staggered_y(mdmc_deltaH, 0), color='red', marker='o', 
        alpha=1, label=r'Start MDMC')
    ax[2].scatter(staggered_xf, staggered_y(mdmc_deltaH, n), color='red', marker='x',
        alpha=1, label=r'End MDMC')
    ax[2].grid(True)
    ax[2].legend(loc='best', shadow=True, fontsize = pp.axfont-3)
    # ax[2].set_yscale("log", nonposy='clip')
    ax[2].set_xlabel(r'HMC step, $i$')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, save = False, step_size=0.05, n_steps=20, spacing=1., mixing_angle=np.pi/2., accept_all = False):
    """A wrapper function
    
    Required Inputs
        x0              :: np.array :: initial position input to the HMC algorithm
        pot             :: pot. cls :: defined in hmc.potentials
        file_name       :: string   :: the final plot will be saved with a similar name if save=True
        n_samples       :: int      :: number of HMC samples
        n_burn_in       :: int      :: number of burn in samples
    
    Optional Inputs
        save :: bool    :: bool     :: True saves the plot, False prints to the screen
        n_steps         :: int      :: LF trajectory lengths
        step_size       :: int      :: Leap Frog step size
        spacing         :: float    :: lattice spacing
    """
    
    model = Model(x0, pot, step_size=step_size, n_steps=n_steps, spacing=spacing)
    model.sampler.accept.store_acceptance = True
    model.sampler.dynamics.save_path = True
    model.sampler.accept.accept_all = accept_all
    
    print 'Running Model: {}'.format(file_name)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True, 
    mixing_angle=mixing_angle)
    print 'Finished Running Model: {}'.format(file_name)
    
    #### This is pulling data from the Metropolis-Hastings Steps
    # pull out all the data from the metropolis class
    accept_rates = np.asarray(model.sampler.accept.accept_rates[n_burn_in:]).flatten()
    accept_rejects = np.asarray(model.sampler.accept.accept_rejects[n_burn_in:]).flatten()
    exp_delta_hs = np.asarray(model.sampler.accept.exp_delta_hs[n_burn_in:]).flatten()
    h_olds = np.asarray(model.sampler.accept.h_olds[n_burn_in:]).flatten()
    h_news = np.asarray(model.sampler.accept.h_news[n_burn_in:]).flatten()
    
    # get the means where relevent
    av_acc      = np.asscalar(accept_rates.mean())
    meas_av_acc = np.asscalar(accept_rejects.mean())
    meas_av_exp_dh  = np.asscalar(exp_delta_hs.mean())
    
    ### This is pulling data from the MDMC steps
    p_mdmc = model.sampler.dynamics.p_ar # contains start step from each previous
    x_mdmc = model.sampler.dynamics.x_ar # so shape is n_steps + 1
    
    # these are both lists so need to vonvert back to ararys
    kE_mdmc = map(model.sampler.potential.kE, p_mdmc) # calculate kE
    uE_mdmc = map(model.sampler.potential.uE, x_mdmc) # calculate uE
    h_mdmc = np.asarray(np.asarray(kE_mdmc) + np.asarray(uE_mdmc))
    
    # filters out all the values multiples of n_steps
    h_mdmc_0 = np.repeat(h_mdmc[0::(n_steps+1)], 11)
    delta_H = h_mdmc - h_mdmc_0
    
    print '\n\t<Prob. Accept>: {:4.2f}'.format(av_acc)
    print '\t<Prob. Accept>: {:4.2f}     (Measured)'.format(meas_av_acc)
    print '\t<exp{{-𝛿H}}>:     {:8.2E} (Measured)\n'.format(meas_av_exp_dh)
    print '\tstep size:      {:4.2f}'.format(model.sampler.dynamics.step_size)
    print '\ttraj. length:   {:4.2f}'.format(model.sampler.dynamics.n_steps)
    
    pow_ten = int(np.floor(np.log10(meas_av_exp_dh)))   # calculate N for AeN sci. format
    decimal = meas_av_exp_dh / float(10**pow_ten)       # calculate A for AeN
    
    # one long subtitle - long as can't mix LaTeX and .format()
    subtitle = 'Potential: {}, Lattice shape: {}'.format(pot.name, x0.shape) + \
        r', $\langle P_{\text{acc}} \rangle= '+' {:4.2f}$'.format(meas_av_acc) + \
        r', $\langle e^{-\delta H} \rangle='+' {:4.2f}'.format(decimal)+r'\times 10^{'\
            + '{}'.format(pow_ten) + '}$'
    
    plot(accept_rates, accept_rejects, h_olds, h_news, exp_delta_hs,
        h_mdmc, np.exp(-delta_H),
        subtitle = subtitle,
        save = saveOrDisplay(save, file_name)
        )
    pass