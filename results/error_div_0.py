#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC
from common.utils import saveOrDisplay

save = False

pp = Pretty_Plotter()
pp._teXify() # LaTeX
pp.params['text.latex.preamble'] = r"\usepackage{amsmath}"
pp._updateRC()

fig = plt.figure(figsize=(8, 8)) # make plot
ax =[]
ax.append(fig.add_subplot(111))

ax[0].set_title(r'Checking stability of $g_W(\tau_W)$ for $\tau_W = \lim_{g_{\text{int}}\to 0} \left[\ln\left(1+\frac{1}{g_{\text{int}}}\right)\right]^{-1}$', fontsize=pp.ttfont)

ax[0].set_ylabel(r'$g_W(\tau_W) = e^{-1/\tau_W} - \tau_W$')
ax[0].set_xlabel(r'$g_{\text{int}}$')

x = np.linspace(-.2, .1, 1000)
ax[0].plot(x, np.exp(-1./np.log(1. + 1./x.clip(0))) - x, label=r'$1/0$ = np.inf', linewidth=2., alpha = .4)
ax[0].plot(x, np.exp(-1./np.log(1. + 1./x.clip(np.finfo(float).eps))) - x, label=r"$1/0$ = `eps'", linewidth=2., alpha = .4)

ax[0].legend(loc='best', shadow=True, fontsize = pp.axfont)
pp.save_or_show(saveOrDisplay(save, __file__), PLOT_LOC)