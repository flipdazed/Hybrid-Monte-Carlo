import numpy as np
import matplotlib.pyplot as plt

from models import Basic_HMC as Model
from utils import saveOrDisplay
from plotter import Pretty_Plotter, PLOT_LOC

def plot(samples, subtitle, y_label, save, extra_data=[]):
    """Note that samples and burn_in contain the initial conditions
    
    Required Inputs
        samples     :: the 1D numpy array to plot as a histogram
        subtitle    :: str :: subtitle for the plot
        y_label     :: str  :: the y axis label
        save        :: bool :: True saves the plot, False prints to the screen
    
    Optional Inputs
        extra_data :: list of dicts :: see below
    
    Expectations
        extra_data = [{'f':actual, 'label':theory},
                      {'f':fitted, 'label':None}]
        'f' contains a function with respect to a linear x range (x-axis) and the samples
        f(x, samples)
        'label' is a string or None
        
    """
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize = (8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    # burn_in includes initial cond.
    # samples inclues final burn_in as initial cond.
    fig.suptitle(subtitle,
        fontsize=16)
    ax[0].set_title(r'{} HMC samples'.format(samples.shape[0]), fontsize=pp.ttfont-4)
    
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel(r"Position, $x$")
    
    n, bins, patches = ax[0].hist(samples.ravel(), 50, normed=1, # histogram
        facecolor='green', alpha=0.2, label=r'Sampled data')
    
    n = 100 # size of linear space
    x = np.linspace(-5, 5, n)
    for i in extra_data: 
        params, fitted = i['f'](x, samples)
        ax[0].plot(x, fitted, 
            linewidth=2., alpha=0.6, label=i['label'].format(*params))
    
    ax[0].legend(loc='best', shadow=True, fontsize = pp.axfont)
    ax[0].grid(False)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def main(x0, pot, file_name, n_samples, n_burn_in, y_label, extra_data = [], save = False):
    """A wrapper function
    
    Required Inputs
        x0          :: np.array :: initial position input to the HMC algorithm
        pot         :: potential class :: defined in hmc.potentials
        file_name   :: string :: the final plot will be saved with a similar name if save=True
        n_samples   :: int :: number of HMC samples
        n_burn_in   :: int :: number of burn in samples
        y_label     :: str :: the y axis label
        
    Optional Inputs
        extra_data :: list of dicts :: see below
        save :: bool :: True saves the plot, False prints to the screen
    
    Expectations
        extra_data = [{'f':actual,'x':x, 'label':theory},
                      {'f':fitted,'x':x, 'label':None}]
        'f' contains a function with respect to a linear x range (x-axis)
        'label' is a string or None
    """
    
    model = Model(x0, pot=pot)
    print 'Running Model: {}'.format(file_name)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
    print 'Finished Running Model: {}'.format(file_name)
    
    # first contains the last burn in sample
    samples = np.asarray(model.samples)[1:]
    
    plot(samples = samples, 
        y_label = y_label,
        subtitle = r'Potential: {}, Lattice shape: {}'.format(pot.name, x0.shape),
        extra_data = extra_data,
        save = saveOrDisplay(save, file_name)
        )