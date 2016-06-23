import numpy as np
import os
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from hmc_qho_1d import Model

def plot(burn_in, samples, save='hmc_qho_1d_lfpaths.png'):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize = (8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    name = "QHO"
    fig.suptitle(r'HMC path configurations, sampling the {} potential'.format(name),
        fontsize=16)
    
    ax[0].set_title(
        r'{} Burn-in configurations shown in grey'.format(burn_in.shape[0]-1))
    
    ax[0].set_ylabel(r'Time, $\tau$')
    ax[0].set_xlabel(r"Position, $x(\tau)$")
    
    for i in xrange(burn_in.shape[0]):
        offst = burn_in[i].size + 1 # burn-in samples
        ax[0].plot(burn_in[i], np.arange(1, offst),
        linestyle='--', alpha=.2, linewidth=3, color='grey')
    
    for i in xrange(samples.shape[0]):
        offst = samples[i].size + 1 # burn-in samples
        ax[0].plot(samples[i], np.arange(1, offst),
        linestyle='-', alpha=.3, linewidth=3, color='green')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    n_burn_in = 5
    n_samples = 5
    
    model = Model()
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    burn_in = model.burn_in
    samples = model.samples
    
    plot(burn_in[:n_burn_in+1], samples=samples,
        # save=save_name
        save=False
    )