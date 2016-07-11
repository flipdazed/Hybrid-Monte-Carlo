import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from correlations import acorr
from models import Basic_HMC as Model
from utils import saveOrDisplay
from data import store
from plotter import Pretty_Plotter, PLOT_LOC

def plot(ac_fn, subtitle, save, theory=None):
    """Plots the two-point correlation function
    
    Required Inputs
        ac_fn :: np.array :: correlation function
        subtitle :: string :: subtitle for the plot
        save :: bool :: True saves the plot, False prints to the screen
        theory :: none atm
    
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
    
    fig.suptitle(r"Autocorrelation Function for $\hat{x}$",
        fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'HMC trajectories, $\tau / n\delta\tau$')
    ax[0].set_ylabel(r'$\langle (x_i - \bar{x})(x_{i+t} - \bar{x}) \rangle$')
    
    steps = np.linspace(0, ac_fn.size, ac_fn.size, False)    # get x values
    
    log_y = np.log(ac_fn)        # regress for logarithmic scale
    mask = np.isfinite(log_y)   # handles negatives in the logarithm
    x = steps[mask]
    y = log_y[mask]
    # linear regression of the exponential curve
    m, c, r_val, p_val, std_err = stats.linregress(x, y)
    fit = np.exp(m*steps + c)
    
    if theory is not None:
        th = ax[0].plot(steps, theory, color='green',linewidth=3., linestyle = '-', 
            alpha=0.2, label = r'Theoretical prediction', marker='o')
    
    # exponential regression (straight line on log scale)
#    bf = ax[0].plot(steps, fit, color='blue',
#         linewidth=3., linestyle = '-', alpha=0.2, label=r'fit: $y = e^{'\
#             + '{:.2f}x'.format(m)+'}e^{'+'{:.2f}'.format(c) + r'}$')
    
    f = ax[0].stem(steps, ac_fn, markerfmt='ro', linefmt='k:', basefmt='k-',
        label=r'MCMC Data')
    
    ax[0].set_xlim(xmin=-1)
#    ax[0].set_yscale("log", nonposy='clip')
    
    ax[0].legend(loc='best', shadow=True, fontsize = pp.axfont)
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, c_len=20, step_size = .5, n_steps = 20, spacing = 1., save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
    
    Optional Inputs
        c_len :: int :: length of autocorellation
        step_size :: float :: MDMC step size
        n_steps :: int :: number of MDMC steps
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    
    rng = np.random.RandomState()
    model = Model(x0, pot=pot, spacing=spacing, rng=rng, step_size = step_size,
      n_steps = n_steps)
    c = acorr.Autocorrelations_1d(model, 'run', 'samples')
    
    subtitle = r"Potential: {}; Lattice Shape: ${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        pot.name, x0.shape, spacing, step_size, n_steps)
    length = model.x0.size
    
    
    print 'Running Model: {}'.format(file_name)
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
    ac_fn = np.asarray([c.getAcorr(separation=i) for i in range(c_len)])
    print 'Finished Running Model: {}'.format(file_name)
    
    av_x0_sq = ac_fn[0]
    print 'measured: <x(0)x(0)> = {}'.format(av_x0_sq)
    store.store(ac_fn, file_name, '_acfn')
    plot(ac_fn, subtitle, 
        save = saveOrDisplay(save, file_name))
    
