import numpy as np
from scipy.signal import residuez
import matplotlib.pyplot as plt

from plotter import Pretty_Plotter, PLOT_LOC
from common.utils import saveOrDisplay
from theory.autocorrelations import M2_Fix, M2_Exp
__doc__ = """::References::
[1] Cost of the Generalised Hybrid Monte Carlo Algorithm for Free Field Theory

The attempt here is to define x^n = \exp(-n \beta \tau) and determine the roots
for \beta

Firstly I want to determine
"""
tau = 1
m = 1


save      = True
file_name = __file__

def plot(lines, subtitles, save):
    """
    Required Inputs
        lines       :: dict :: axis(int): (x(np.ndarray), y(np.ndarray), label(str))
        subtitles   :: dict :: axis(int): subtitle(str)
        save        :: bool/str :: string saves the plot as given string, False prints to the screen
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()
    
    fig, ax = plt.subplots(2, sharex=True, figsize = (8, 8))
    fig.subplots_adjust(hspace=0.2)
    
    fig.suptitle(r"Comparing methods of calculating $\mathcal{L}^{-1}F(\beta)$", fontsize=pp.ttfont)
    
    for i, subtitle in subtitles.iteritems(): 
        ax[i].set_title(subtitle, fontsize=pp.ttfont-4)
    
    ax[-1].set_xlabel(r'Fictitious sample time, $t = \sum^j_{i=0}\tau_i \stackrel{j\to\infty}{=} j\bar{\tau} = j \delta \tau \bar{n}$')
    
    for i, lines in lines.iteritems():
        for w, line in enumerate(lines):
            x, y, label = line
            style = '--' if not w%2 else '-'
            ax[i].plot(x, y, label=label, linewidth=(w+1)*2, alpha=0.4, linestyle=style)
    
    for i in ax: 
        i.grid(True) 
        i.legend(loc='best', shadow=True, fontsize = pp.tfont, fancybox=True)
        i.set_ylabel(r'Autocorrelation, $C(t)$') # same y axis for all
    
    pp.save_or_show(save, PLOT_LOC)
    pass

if __name__ == '__main__':
    mf = M2_Fix(tau, m)
    me = M2_Exp(tau, m)
    
    print
    print 'Checking Root and Pole finding...'
    print 'Fixed Poles',mf.poles
    print 'Fixed Residues',mf.res
    print 'Fixed Constant',mf.const
    print
    print 'Exp Poles',me.poles
    print 'Exp Residues',me.res
    print 'Exp Constant',me.const
    print
    
    t = np.linspace(0,10,1000)
    fE = lambda t: np.real(np.asarray([a_i*np.exp(b_i*t) for a_i, b_i in zip(me.res, me.poles)]).sum(0))
    
    mu=np.cos(m*tau)**2  
    r,p,k = residuez([mu,0], [1,-np.cos(m*tau)**2])
    fF = lambda t: np.real(np.sum(k) + np.asarray([a_i/b_i*b_i**(t/tau) for a_i, b_i in zip(r,p)]).sum(0))
    
    lines = {
        0:[(t, fE(t), r'\verb|scipy.signal.residuez()|'),(t, me.eval(t), r'Analytical partial fractioning')],
        1:[(t, fF(t), r'\verb|scipy.signal.residuez()|'),(t, mf.eval(t), r'Analytical partial fractioning')]
        }
    
    app = r"$\tau={:4.2f};m={:3.1f}$".format(tau, m)
    subtitles = {
        0:"Exponentially Distributed Trajectories "+app, 
        1:"Fixed Trajectories "+app
        }
    
    plot(lines, subtitles, 
        save=saveOrDisplay(save, file_name)
        )