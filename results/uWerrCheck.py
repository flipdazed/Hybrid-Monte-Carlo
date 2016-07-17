# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import json

from plotter import Pretty_Plotter, PLOT_LOC
from common.utils import saveOrDisplay
from correlations.errors import uWerr

save      = True
file_name = __file__

def plot(res, actual, save):
    """Plots 2 stacked figures:
        1. The ratio of changes in integrated autocorrelation time
        2. The ratio of changes in autocorrelation time
    
    Required Inputs
        probs           :: dict :: as below
        accepts         :: dict :: as below
        save            :: bool :: True saves the plot, False prints to the screen
    """
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp._updateRC()

    fig, ax = plt.subplots(2, figsize = (8, 8))
    fig.suptitle(r'Comparisions of \verb|uWerr()| in \verb|python|:$A[X]$ and \verb|matlab|:$U[X]$',
        fontsize=pp.ttfont)
    ax[0].set_xlabel(r'Integration Window, $w$', fontsize=pp.tfont)

    x = np.arange(actual['itau_aav'].size)

    ax[0].set_title(r'Measuring the ratio $\Delta = 1 - \frac{U[X]}{A[X]}$', fontsize = pp.tfont)
    ax[0].plot(x, (res['itau_aav']-actual['itau_aav'])/res['itau_aav'], label=r'$X = \bar{\bar{\tau}}(w)$')
    ax[0].set_ylabel(r'$\Delta$', fontsize=pp.tfont-2)
    ax[0].legend(loc='best', shadow=True, fontsize = pp.tfont, fancybox=True)

    diff_acorr = res['acorr']-actual['acorr']
    ax[1].plot(x, diff_acorr/res['acorr'], label=r"$X = \Gamma(t)$")
    ax[1].set_ylabel(r"$\Delta$", fontsize=pp.tfont-2)
    ax[1].legend(loc='best', shadow=True, fontsize = pp.tfont, fancybox=True)
    ax[1].set_xlabel(r'Autocorrelation Time, $t$', fontsize=pp.tfont)
    plt.show()
    
    for i in ax: i.grid(True) 
    pp.save_or_show(save, PLOT_LOC)
    pass


if __name__ == '__main__':
    d = './correlations/ref_code/' # grab uWerr data from Matlab script
    with open(d+'/uWerr_out.dat') as f: actual = json.load(f)
    actual['itau_aav'] = np.loadtxt(d + 'uWerr_out_tauintFbb.dat')
    actual['acorr'] = np.loadtxt(d + 'uWerr_out_gammaFbb.dat')
    a = np.loadtxt(d+'actime_tint20_samples.dat')
    # run the uWerr from here
    value, dvalue, ddvalue, tauint, dtauint, y, y2 = uWerr(a)
    res = {'value':value,
        'dvalue':dvalue,
        'ddvalue':ddvalue,
        'dtauint':dtauint,
        'tauint':tauint,
        # 'CFbbopt':c_aav,
        'itau_aav':y,
        'acorr':y2}

    w = np.around((ddvalue/dvalue)**2*a.size - .5, 0)
    w2 = np.around((dtauint/tauint/2.)**2*a.size - .5 + tauint, 0)
    assert w == w2 # sanity check

    for k in sorted(res.keys()):
        print 'Param:{:8s} my code:{:8.4E}, UF:{:8.4E}, change: {:6.5f}%'.format(k, np.average(res[k]), np.average(actual[k]), np.average((res[k]-actual[k])/res[k]*100.))

    plot(res, actual, save=saveOrDisplay(save, file_name))