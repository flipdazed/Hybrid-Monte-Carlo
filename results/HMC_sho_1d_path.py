import numpy as np
import os
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC

from HMC_sho_1d import Model

def plot(burn_in, samples, save='HMC_sho_1d_path.png'):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig = plt.figure(figsize = (8, 8)) # make plot
    ax =[]
    ax.append(fig.add_subplot(111))
    
    name = "SHO"
    fig.suptitle(r'Example HMC path sampling the {} potential'.format(name),
        fontsize=16)
    
    ax[0].set_title(
        r'{} Burn-in Samples shown in orange'.format(burn_in.size-1))
    
    ax[0].set_ylabel(r'Sample, $n$')
    ax[0].set_xlabel(r"Position, $x$")
    
    # burn_in includes initial cond.
    # samples inclues final burn_in as initial cond.
    
    offst = burn_in.size # burn-in samples
    ax[0].plot(burn_in, np.arange(0, offst), #marker='x',
        linestyle='-', color='orange', label=r'Burn In')
    
    ax[0].plot(samples, np.arange(offst-1, offst-1 + samples.size), #marker='x',
        linestyle='-', color='blue', label=r'Sampling')
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    n_burn_in = 10
    n_samples = 25
    
    model = Model()
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    
    burn_in = np.asarray(model.burn_in).reshape(n_burn_in+1)
    samples = np.asarray(model.samples).reshape(n_samples+1)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    plot(burn_in[:n_burn_in+1], samples=samples, 
        save=save_name
        # save=False
    )