import numpy as np
import os
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from HMC_qho_1d import Model
from scipy.stats import norm

def plot(samples, save='HMC_qho_1d_pot.py'):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize = (8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    name = "QHO"
    fig.suptitle(r'Sampled {} Potential'.format(name), fontsize=16)
    
    ax[0].set_title(
        r'{} HMC samples'.format(samples.shape[0]-1))
    
    ax[0].set_ylabel(r'Sampled Potential, $e^{-V(x)}$')
    ax[0].set_xlabel(r"Position, $x$")
    
    n = 100 # size of linear space
    x = np.linspace(-5, 5, n)
    
    p = norm.fit(samples.ravel())
    fitted = norm.pdf(x, loc=p[0], scale=p[1])
    
    w = np.sqrt(1.25) # w  = 1/(sigma)^2
    sigma = 1./np.sqrt(2*w)
    c = np.sqrt(w / np.pi) # this is the theory
    actual = np.exp(-x**2*1.1)*c
    
    theory = r'$|\psi(x)|^2 = \sqrt{\frac{\omega}{\pi}}e^{-\omega x^2}$ for $\omega=\sqrt{\frac{5}{4}}$'
    
    n, bins, patches = ax[0].hist(samples.ravel(), 50, normed=1, # histogram
        facecolor='green', alpha=0.2, label=r'Sampled Data')
        
    ax[0].plot(x, fitted, # marker='x', # best fit
        linestyle='-', color='red', linewidth=2., alpha=0.6,
        label=r'Fitted Potential')
    
    ax[0].plot(x, actual, # marker='x',
        linestyle='--', color='blue', linewidth=2., alpha=0.6,
        label=r'Theory: {}'.format(theory))
    
    ax[0].legend(loc='upper left', shadow=True, fontsize = pp.axfont)
    ax[0].grid(False)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    n_burn_in = 5
    n_samples = 100
    
    model = Model()
    print 'Running Model'
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    print 'Finished Running Model'
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    burn_in = model.burn_in
    samples = model.samples
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    plot(samples=samples,
        save=save_name
        # save=False
    )
