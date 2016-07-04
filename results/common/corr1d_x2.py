import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from correlations import corr
from models import Basic_HMC as Model
from utils import saveOrDisplay

from plotter import Pretty_Plotter, PLOT_LOC

def plot(c_fn, spacing, subtitle, save):
    """Plots the two-point correlation function
    
    Required Inputs
        c_fn :: np.array :: correlation function
        spacing :: float :: the lattice spacing
        subtitle :: string :: subtitle for the plot
        save :: bool :: True saves the plot, False prints to the screen
    
    Optional Inputs
        all_lines :: bool :: if True, plots hamiltonian as well as all its components
    """
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = r"\usepackage{amsmath}"
    pp._updateRC()
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    fig.suptitle(r"Two-Point Correlation Function, $\langle x(0)x(\tau)\rangle$",
        fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Time Separation, $\tau$')
    ax[0].set_ylabel(r'$\langle x(0)x(\tau) \rangle$')
    
    steps = np.linspace(0, spacing*c_fn.size, c_fn.size, False)    # get x values
    log_y = np.log(c_fn)        # regress for logarithmic scale
    mask = np.isfinite(log_y)   # handles negatives in the logarithm
    masked_x = steps[mask]
    # linear regression of the exponential curve
    masked_fit = np.exp(np.poly1d(np.polyfit(masked_x, log_y[mask], 1))(masked_x))
    
    bf = ax[0].plot(masked_x, masked_fit, color='blue',
         linewidth=5., linestyle = '-', alpha=0.2)
    f = ax[0].scatter(steps, c_fn, color='red', marker='x')
    ax[0].set_xlim(xmin=0)
    ax[0].set_yscale("log", nonposy='clip')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, spacing = 1., save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        
    Optional Inputs
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    
    rng = np.random.RandomState()
    model = Model(x0, pot=pot, spacing=spacing, rng=rng)
    c = corr.Corellations_1d(model, 'run', 'samples')
    
    subtitle = r"Potential: {}; Lattice Shape: ${}$; $a={:.1f}$".format(
        pot.name, x0.shape, spacing)
    length = model.x0.size
    if hasattr(pot, 'mu'):
        m0 = pot.m0
        mu = pot.mu
        th_x_sq = corr.qho_theory(spacing, mu, length)
        print 'theory:   <x(0)x(0)> = {}'.format(th_x_sq)
        subtitle += r"; $m_0={:.1f}$; $\mu={:.1f}$;".format(m0, mu)
    else:
        m0, mu = None
    
    print 'Running Model: {}'.format(file_name)
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
    c_fn = np.asarray([c.twoPoint(separation=i) for i in range(5)])
    print 'Finished Running Model: {}'.format(file_name)
    
    av_x0_sq = c_fn[0]
    print 'measured: <x(0)x(0)> = {}'.format(av_x0_sq)
    plot(c_fn, spacing, subtitle, 
        save = saveOrDisplay(save, file_name))
    