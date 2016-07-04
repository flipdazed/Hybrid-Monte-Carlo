import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from models import Basic_HMC as Model
from utils import saveOrDisplay

from plotter import Pretty_Plotter, PLOT_LOC

def plot(norm, subtitle, save):
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    fig.suptitle(r'Magnitude of Change in Phase Space, $\Delta\mathcal{P}(x,p)$',
        fontsize=pp.ttfont)
    
    ax[0].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Integration Step, $n$')
    ax[0].set_ylabel(r"$|\Delta\mathcal{P}(x,\pi)| = \sqrt{(\pi_{t} + \pi_{\text{-}t})^2 + (x_{t} - x_{\text{-}t})^2}$")
    
    n_steps = norm.size - 1 # norm contains 0th step
    steps = np.linspace(0, n_steps, n_steps+1, True)
    
    ax[0].plot(steps, norm, linestyle='-', color='blue')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def reverseIntegration(p0, x0, model, n_steps, progress_bar):
    """Reverses the integration after n_steps
    and returns the position and momentum paths
    during the integration: returns (p, x)
    
    Required Inputs
        model   :: the model class  :: as an instance
        x0      :: lattice or c_vec :: initial position config
        p0      :: lattice or c_vec :: initial momentum config
        n_steps :: int              :: number of integration steps
        progress_bar :: bool        :: show a progress bar if True
    """
    
    # start the dynamics integrator
    # note the strange flips necessary are because 
    # the momentum flip is not included in the Leap Frog routine
    model.dynamics.newPaths()
    if progress_bar: print "Forwards integration..."
    pm, xm = model.dynamics.integrate(copy(p0), copy(x0), verbose = progress_bar)
    if progress_bar: print "Reverse integration..."
    p0f, x0f = model.dynamics.integrate(-pm, xm, verbose = progress_bar) # time flip
    p0f = -p0f # time flip to point in right time again
    
    # extract the paths
    p_path, x_path = model.dynamics.p_ar, model.dynamics.x_ar
    
    p_path = np.asarray(p_path)
    x_path = np.asarray(x_path)
    
    # curious why I need to clip one step on each?
    # something to do with the way I sample the steps...
    # clip last step on forward and first step on backwards
    # solved!... because I didn't save the zeroth step in the integrator
    # integrator nowsaves zeroth steps
    mid = p_path.shape[0]//2
    change_p = (-p_path[ : mid] - p_path[mid : ][::-1])**2
    change_x = ( x_path[ : mid] - x_path[mid : ][::-1])**2
    norm = np.sqrt(change_p + change_x)
    
    return norm
#
def main(x0, pot, file_name, save = False, n_steps = 500, step_size = 0.05, progress_bar = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
    
    Optional Inputs
        save :: bool :: True saves the plot, False prints to the screen
        n_steps :: list :: LF trajectory lengths
        step_size :: float :: Leap Frog step size
        progress_bar :: bool        :: show a progress bar if True
    """
    # set up the model
    model = Model(x0,
        pot       = pot,
        n_steps   = n_steps, 
        step_size = step_size
        )
    model.dynamics.save_path = True
    
    # initial conditions - shoudn't matter much
    p0 = model.sampler.p
    x0 = model.sampler.x
    
    print 'Running Model: {}'.format(file_name)
    norm = reverseIntegration(p0, x0, model, n_steps, progress_bar)
    print 'Finished Running Model: {}'.format(file_name)
    
    # average change across all sites
    av_norm = norm.mean(axis=1)
    
    plot(av_norm,
        subtitle = r'Potential: {}, Lattice shape: {}'.format(pot.name, x0.shape),
        save = saveOrDisplay(save, file_name)
        )
    pass