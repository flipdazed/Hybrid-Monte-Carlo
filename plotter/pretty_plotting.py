# Begin my own plotter class
from matplotlib import rcParams
import matplotlib.pyplot as plt

import subprocess

class Pretty_Plotter(object):
    """contains things to make plots look nice"""
    def __init__(self):
        pass
    
    def _teXify(self):
        """makes plots look posh"""
        
        self.s = 1   # Increase plot size by a scale factor
        self.fig_dims = [12*self.s,5*self.s]    # size of plot
        self.axfont = 11*self.s                 # axes
        self.tfont  = 14*self.s                 # subplot titles
        self.ttfont = 16*self.s                 # figure title
        
        # Customising Options
        self.params = {'text.usetex' : True,
                  'font.size' : 11,
                  'font.family' : 'lmodern',
                  'text.latex.unicode': True,
                  'text.latex.preamble': [r"\usepackage{lmodern}"],
                  # 'text.latex.preamble': [r"\usepackage{hyperref}"], # doesn't work
                  # 'text.latex.preamble': [r"\usepackage{amsmath}"],
                  'figure.figsize' : self.fig_dims,
                  'figure.subplot.top':    0.90, #0.85 for title
                  'figure.subplot.hspace': 0.40,
                  'figure.subplot.wspace': 0.40,
                  'figure.subplot.bottom': 0.15,
                  'axes.titlesize': self.tfont,
                  'axes.labelsize': self.axfont,
                  'axes.grid': True
                  }
              
        self._updateRC()
        pass
    
    def _updateRC(self):
        """updates with new poshness"""
        rcParams.update(self.params) # updates the default parameters
        pass
    
    def save_or_show(self, save, d):
        """Will either print to screen or save to location save
        
        Required Inputs
            save :: string / bool :: save location or '' / False
            d    :: string :: directory to save in if save != False
        """
        if save:
            save_dir = d
            subprocess.call(['mkdir', d])
            
            fig = plt.gcf()
            fig.savefig(save_dir+save)
        else:
            plt.show()
        pass