import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from correlations import corr
from models import Basic_HMC as Model
from utils import saveOrDisplay

from plotter import Pretty_Plotter, PLOT_LOC

def plot(c_fn, theory, spacing, subtitle, save):
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
    x = steps[mask]
    y = log_y[mask]
    # linear regression of the exponential curve
    m, c, r_val, p_val, std_err = stats.linregress(x, y)
    fit = np.exp(m*steps + c)
    
    if theory is not None:
        th = ax[0].plot(steps, theory, color='green',linewidth=3., linestyle = '-', 
            alpha=0.2, label = r'Theoretical prediction', marker='o')
    
    bf = ax[0].plot(steps, fit, color='blue',
         linewidth=3., linestyle = '-', alpha=0.2, label=r'fit: $y = e^{'\
             + '{:.2f}x'.format(m)+'}e^{'+'{:.2f}'.format(c) + r'}$')
    f = ax[0].scatter(steps, c_fn, color='red', marker='x',
        label=r'MCMC Data')
    
    ax[0].set_xlim(xmin=0)
    ax[0].set_yscale("log", nonposy='clip')
    
    ax[0].legend(loc='best', shadow=True, fontsize = pp.axfont)
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, c_len=5, step_size = .5, n_steps = 50, spacing = 1., save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        
    Optional Inputs
        c_len :: int :: length of corellation
        step_size :: float :: MDMC step size
        n_steps :: int :: number of MDMC steps
        spacing ::float :: lattice spacing
        save :: bool :: True saves the plot, False prints to the screen
    """
    
    rng = np.random.RandomState()
    model = Model(x0, pot=pot, spacing=spacing, rng=rng, step_size = step_size,
      n_steps = n_steps)
    c = corr.Corellations_1d(model, 'run', 'samples')
    
    subtitle = r"Potential: {}; Lattice Shape: ${}$; $a={:.1f}$".format(
        pot.name, x0.shape, spacing)
    length = model.x0.size
    
    if hasattr(pot, 'mu'):
        m0 = pot.m0
        mu = pot.mu
        th_x_sq = np.asarray(
            [corr.qho_theory(spacing, mu, length, i) for i in range(c_len)])
        print 'theory:   <x(0)x(0)> = {}'.format(th_x_sq[0])
        subtitle += r"; $m_0={:.1f}$; $\mu={:.1f}$;".format(m0, mu) \
                    + r' $N_{\text{HMC}}' + '={}$'.format(n_samples)
    else:
        th_x_sq, m0, mu = None
    
    print 'Running Model: {}'.format(file_name)
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
    c_fn = np.asarray([c.twoPoint(separation=i) for i in range(c_len)])
    print 'Finished Running Model: {}'.format(file_name)
    
    av_x0_sq = c_fn[0]
    print 'measured: <x(0)x(0)> = {}'.format(av_x0_sq)
    plot(c_fn, th_x_sq, spacing, subtitle, 
        save = saveOrDisplay(save, file_name))
    
