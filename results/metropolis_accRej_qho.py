#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import os
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from common.hmc_model import Model
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

def plot(probs, save):
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize = (8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    name = "SHO"
    fig.suptitle(r'Sampled {} Ground State Potential'.format(name), fontsize=16)
    
    ax[0].set_title(
        r'Acceptance rates from {} HMC samples'.format(len(probs)))
    
    ax[0].set_ylabel(r'Probability')
    ax[0].set_xlabel(r"HMC step")
    
    ax[0].plot(probs, # marker='x',
        linestyle='-', color='blue', linewidth=2., alpha=0.6,
        # label=r'Theory: {}'.format(theory)
        )
    
    # ax[0].legend(loc='upper left', shadow=True, fontsize = pp.axfont)
    ax[0].grid(True)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    n_burn_in = 5
    n_samples = 50
    
    dim = 1; n = 10
    x0 = np.random.random((n,)*dim)
    
    pot = QHO()
    model = Model(x0, pot = pot, step_size=0.1, n_steps=5, spacing=1.)
    model.sampler.accept.store_acceptance = True
    
    print 'Running Model: {}'.format(__file__)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    print 'Finished Running Model: {}'.format(__file__)
    
    accept_rates = np.asarray(model.sampler.accept.accept_rates).ravel()
    accept_rejects = np.asarray(model.sampler.accept.accept_rejects).ravel()
    delta_hs = np.asarray(model.sampler.accept.delta_hs).ravel()
    
    print '\n\t<Prob. Accept>: {:4.2f}'.format(accept_rates.mean(axis=0))
    print '\t<Prob. Accept>: {:4.2f}     (Measured)'.format(accept_rejects.mean(axis=0))
    print '\t<exp{{-𝛿H}}>:     {:8.2E} (Measured)\n'.format(delta_hs.mean(axis=0))
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    plot(probs=accept_rates,
        save = save_name
        # save = False
        )
