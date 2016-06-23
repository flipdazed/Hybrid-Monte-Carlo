import os
import numpy as np
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from HMC_sho_1d import Model
from scipy.stats import norm
    
def plot(samples, save='hmc_sho_1d_pot.png'):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    n = 100 # size of linear space
    x = np.linspace(-5, 5, n)
    
    fig = plt.figure(figsize = (8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    # burn_in includes initial cond.
    # samples inclues final burn_in as initial cond.
    fig.suptitle(r'Sampled SHO Potential',
        fontsize=16)
    ax[0].set_title(
        r'{} HMC samples. True potential in Blue.'.format(samples.size-1))
    
    ax[0].set_ylabel(r'Sampled Potential, $e^{-V(x)}$')
    ax[0].set_xlabel(r"Position, $x$")
    
    # fitted normal dist. parameters p[0] = mean, p[1] = stdev
    p = norm.fit(samples)
    fitted = norm.pdf(x, loc=p[0], scale=p[1])
    actual = norm.pdf(x)
    
    n, bins, patches = ax[0].hist(samples, 50, normed=1, # histogram
        facecolor='green', alpha=0.5, label=r'Sampled Data')
    
    ax[0].plot(x, fitted, # marker='x', # best fit
        linestyle='-', color='orange', label=r'Fitted Potential')
    
    ax[0].plot(x, actual, # marker='x',
        linestyle='-', color='blue', label=r'True Potential')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    n_burn_in = 100
    n_samples = 1000
    
    model = Model()
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    
    samples = np.asarray(model.samples).reshape(n_samples+1)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    plot(samples=samples,
        save=save_name
        # save=False
        )