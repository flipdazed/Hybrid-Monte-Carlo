# -*- coding: utf-8 -*- 
import numpy as np
import os
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from models import Basic_HMC as Model
from utils import saveOrDisplay

def plot(probs, accepts, h_olds, h_news, exp_delta_hs, subtitle, save):
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
    
    xrng = range(1, probs.size+1)
    for a in ax:
        for x,val in zip(xrng,accepts):
            if val == False: a.axvline(x=x, linewidth=4, color='red', alpha=0.2)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    ax[0].set_ylabel(r'Acceptance Prob., $P_{\text{acc}}$')
    ax[0].plot(xrng, probs, linestyle='-', color='blue', linewidth=2., alpha=0.6)
    ax[0].grid(True)
    
    ax[1].set_ylabel(r'Hamiltonian, $H$')
    ax[1].plot(xrng, h_olds, linestyle='-', color='blue', linewidth=2., alpha=0.4, label=r'$H_{\text{old}}$')
    ax[1].plot(xrng, h_news, linestyle='-', color='green', linewidth=2., alpha=0.6, label=r'$H_{\text{new}}$')
    ax[1].legend(loc='upper left', shadow=True, fontsize = pp.ipfont)
    # ax[1].set_yscale("log", nonposy='clip')
    ax[1].grid(True)
    
    ax[2].set_ylabel(r'$\exp\{-\delta H\}$')
    ax[2].plot(xrng, exp_delta_hs, linestyle='-', color='blue', linewidth=2., alpha=0.6)
    ax[2].grid(True)
    ax[2].set_yscale("log", nonposy='clip')
    ax[2].set_xlabel(r'HMC step, $i$')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, save = False, step_size=0.05, n_steps=20, spacing=1.):
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
    
    print 'Running Model: {}'.format(file_name)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
    print 'Finished Running Model: {}'.format(file_name)
    
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
        subtitle = subtitle,
        save = saveOrDisplay(save, file_name)
        )
    pass
