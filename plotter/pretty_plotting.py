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
        self.axfont = 14*self.s                 # axes
        self.tfont  = 14*self.s                 # subplot titles
        self.ttfont = 14*self.s                 # figure title
        self.ipfont = 14*self.s                  # label fonts
        
        # Customising Options
        self.params = {'text.usetex' : True,
                  'font.size' : 11,
                  'font.family' : 'lmodern',
                  'text.latex.unicode': True,
                  'text.latex.preamble': [
                          r"\usepackage{lmodern}",
                          r"\usepackage{bbold}",
                          r"\usepackage{amsmath}",
                          r"\usepackage{mathrsfs}"
                      ],
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
    
    def save_or_show(self, save, save_dir):
        """Will either print to screen or save to location save
        
        Required Inputs
            save :: string / bool :: save location or '' / False
            d    :: string :: directory to save in if save != False
        """
        if save:
            subprocess.call(['mkdir', save_dir])
            
            fig = plt.gcf()
            fig.savefig(save_dir+save)
        else:
            plt.show()
        pass
    
    def add_label(self, ax, text, fontsize = None, pos = [1, 1]):
        """must have a plot"""
        style = dict(facecolor='white', boxstyle='round')
        x,y = pos
        if fontsize is None: fontsize = self.ipfont
        ax.text(x, y, text,
             horizontalalignment='right',
             verticalalignment='center',
             bbox=style,
             transform = ax.transAxes, fontsize=fontsize)
        pass