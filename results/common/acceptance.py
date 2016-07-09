# -*- coding: utf-8 -*- 
import numpy as np
from scipy.special import j0, jn, erfc
import matplotlib.pyplot as plt

from utils import saveOrDisplay, prll_map
from models import Basic_HMC as Model
from plotter import Pretty_Plotter, PLOT_LOC
from hmc.potentials import Klein_Gordon as KG
from hmc.lattice import Periodic_Lattice, laplacian

__doc__ == """
References
    [1] : ADK, BP, `Acceptances and Auto Correlations in Hybrid Monte Carlo'
    [2] : ADK, BP, `Cost of the generalised hybrid Monte Carlo algorithm for free field theory'
"""


def probLMC1dFree(dtau, m, n):
    """Acceptance probability routines for Langevin Monte Carlo: 
        (HMC with 1 Leap Frog step) in one dimension
         -- for the case of the free field
    
    As defined in [1], Eq. (2.3).
    
    Required Input
        n   :: int   :: number of lattice sites
        dtau :: float :: the Leap Frog step length
        m    :: float :: mass
    """
    sigma = 20. + 18.*m**2 + 6.*m**4 + m**6
    return erfc(.125*dtau**3*np.sqrt(.5*n*sigma))

def probHMC1dFree(tau, dtau, m, lattice_p=np.array([])):
    """Acceptance probability routine for Hybrid Monte Carlo
     -- for the case of the free field
    
    Required Input
        dtau :: float :: the Leap Frog step length
        tau  :: float :: the trajectory length, $n*\delta \tau$
        m    :: float :: mass
    
    Optional Input
        lattice_p :: np.ndarray :: momentum at each lattice site
    
    The massless soln is defined in [1] as $\tau_0$. The subscript
    denoting that this is valid for 0th order 
    Leap Frog integration
    
    Notes
        1. If m == 0: no lattice required
           If m != 0: requires lattice
        3. To clarify: $\tau / \delta\tau = n$ where $n$ is the 
        number of Leap Frog steps for each trajectory
    """
    
    if (m == 0) & (lattice_p.size >= 1000):
        c = 4.*tau
        sigma = .75 - .75*j0(c) + jn(2, c) - .25*jn(4,c)
    else:
        if lattice_p.size == 0:
            raise ValueError('lattice velocities required if m != 0 or n < 1000')
        n = lattice_p.size
        p = lattice_p.ravel()
        a = m**2 + 4*np.sin(np.pi*p/n)**2
        sigma = .125*(a**2*(1. - np.cos(2.*tau*np.sqrt(a)))).mean()
    return erfc(.25*dtau**2*np.sqrt(.5*n*sigma))

def plot(n_rng, prob, theories, subtitle, save):
    """Reproduces figure 1 from [1]
       
    
    Required Inputs
        n_rng           :: np.array :: trajectory lengths
        prob            :: np.array :: acc. probs
        theory          :: np.array :: theoretical acc. probs
        subtitle        :: str      :: the subtitle to put in ax[0].set_title()
        save            :: bool :: True saves the plot, False prints to the screen
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    fig.suptitle(r'Testing Acceptance Rate', fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.tfont)
    ax[-1].set_xlabel(r'Trajectory Length, $\tau=n\delta\tau$')
    
    ### add the lines to the plots
    ax[0].set_ylabel(r'$P_{\text{acc}}$')
    ax[0].scatter(n_rng, prob, marker='x', color='red', label=r'Measured')
    
    for label, theory in theories.iteritems():
        x,y = theory
        ax[0].plot(x, y, linestyle='-', linewidth=2., alpha=0.4, 
            label=label)
    
    ### place legends
    ax[0].legend(loc='best', shadow=True, fontsize = pp.ipfont, fancybox=True)
    
    ### formatting
    for i in ax: i.grid(True) 
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, file_name, n_rng, n_samples=1000, n_burn_in=25, save = False, step_size=0.2):
    """A wrapper function
    
    Required Inputs
        x0              :: np.array :: initial position input to the HMC algorithm
        file_name       :: string   :: the final plot will be saved with a similar name if save=True
        n_rng           :: int arr  :: array of number of leapfrog step sizes
    
    Optional Inputs
        n_samples       :: int      :: number of HMC samples
        n_burn_in       :: int      :: number of burn in samples
        save :: bool    :: bool     :: True saves the plot, False prints to the screen
        step_size       :: int      :: Leap Frog step size
    """
    pot = KG(m=0.)
    
    print 'Running Model: {}'.format(file_name)
    
    f = np.vectorize(lambda t: probHMC1dFree(t, step_size, 0, np.asarray(n_rng)))
    x = np.linspace(0, n_rng[-1]*step_size,101, True)
    theory1 = f(x)
    theories = {r'$p \in [0,n]$': (x, theory1)}
    
    def coreFunc(n_steps):
        """function for multiprocessing support"""
        
        model = Model(x0.copy(), pot, step_size=step_size, n_steps=n_steps)
        model.sampler.accept.store_acceptance = True
        
        prob = 1.
        delta_hs = 1.
        av_dh = -1
        accept_rates = []
        delta_hs = []
        samples = []
        while av_dh < 0:
            model.run(n_samples=n_samples, n_burn_in=n_burn_in)
            accept_rates += model.sampler.accept.accept_rates[n_burn_in:]
            delta_hs += model.sampler.accept.delta_hs[n_burn_in:]
            samples.append(model.samples.copy())
            av_dh = np.mean(delta_hs)
            if av_dh < 0: print 'running again -ve av_dh'
        
        samples = np.concatenate(tuple(samples), axis=0)
        prob = np.mean(accept_rates)
        meas_av_exp_dh  = np.asscalar((1./np.exp(delta_hs)).mean())
        
        t3 = erfc(.5*np.sqrt(av_dh))
        
        return prob, t3
    
    # use multi-core support to speed up
    ans = prll_map(coreFunc, n_rng, verbose=True)
    print 'Finished Running Model: {}'.format(file_name)
    
    prob, theory3 = zip(*ans)
    
    x2 = np.asarray(n_rng)*step_size
    # theories[r'$p_{HMC}$'] = (x2, theory2)
    theories[r'$\text{erfc}(\sqrt{\langle \delta H \rangle}/2)$'] = (x2, theory3)
    
    # one long subtitle - long as can't mix LaTeX and .format()
    subtitle = '\centering Potential: {}, Lattice: {}'.format(pot.name, x0.shape) \
        + r', $\delta\tau = ' + '{:4.2f}$'.format(step_size)
    
    plot(n_rng*step_size, prob,
        theories = theories,
        subtitle = subtitle,
        save = saveOrDisplay(save, file_name),
        )
    pass