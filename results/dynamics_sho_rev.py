import os
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from plotter import Pretty_Plotter, PLOT_LOC

from hmc_sho_1d import Model


def plot(norm, save='dynamics_sho_rev.png'):
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize=(8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    fig.suptitle(r'Magnitude of Change in Phase Space, $\Delta\mathcal{P}(x,p)$',
        fontsize=pp.ttfont)
    
    name = 'SHO'
    ax[0].set_title(r'Potential: {}'.format(name),
        fontsize=pp.ttfont-4)
    
    ax[0].set_xlabel(r'Integration Step, $n$')
    ax[0].set_ylabel(r"$|\Delta\mathcal{P}(x,p)| = \sqrt{(p_{t} + p_{\text{-}t})^2 + (x_{t} - x_{\text{-}t})^2}$")
    
    n_steps = norm.size - 1 # norm contains 0th step
    steps = np.linspace(0, n_steps, n_steps+1, True)
    
    ax[0].plot(steps, norm, linestyle='-', color='blue')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def reverseIntegration(p0, x0, model, n_steps):
    """Reverses the integration after n_steps
    and returns the position and momentum paths
    during the integration: returns (p, x)
    
    Required Inputs
        model   :: the model class  :: as an instance
        x0      :: lattice or c_vec :: initial position config
        p0      :: lattice or c_vec :: initial momentum config
        n_steps :: int              :: number of integration steps
    """
    
    # start the dynamics integrator
    # note the strange flips necessary are because 
    # the momentum flip is not included in the Leap Frog routine
    model.dynamics.newPaths()
    pm, xm = model.dynamics.integrate(copy(p0), copy(x0))
    p0f, x0f = model.dynamics.integrate(-pm, xm) # time flip
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
if __name__ == '__main__':
    
    n_steps = 5000000
    
    # set up the model
    model = Model()
    model.dynamics.save_path = True
    model.dynamics.n_steps = n_steps
    
    # initial conditions - shoudn't matter much
    p0 = np.asarray([[4.]])
    x0 = np.asarray([[1.]])
    
    print 'Running Model'
    norm = reverseIntegration(p0, x0, model, n_steps)
    print 'Finished Running Model'
    
    # average change across all sites
    av_norm = norm.mean(axis=1)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    plot(av_norm, 
        # save=False
        save=save_name
        )