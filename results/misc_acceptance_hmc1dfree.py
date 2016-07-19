#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from PIL import Image

from data import store
from plotter import Pretty_Plotter, PLOT_LOC
from common.acceptance import probHMC1dFree as prob

plt.ion()
pp = Pretty_Plotter()
pp._teXify() # LaTeX
pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
pp._updateRC()

dx = (4-2)/5.
dy = (.98-.96)/5.

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

ax.set_title(r'Rescaling in $x$ required to match theory: $\delta \tau $ vs. $\delta \tau/2$', fontsize=pp.ttfont)
ax.set_xlabel(r'$\tau_0 = n\delta\tau$', fontsize=pp.tfont)
ax.set_ylabel(r'$\langle P_{\text{acc}} \rangle$', fontsize=pp.tfont)

xlim = (0-.5*dx, 8+.5*dx)
ylim = (.96-2.5*dy, 1.+1.5*dy)
xloc = plticker.MultipleLocator(base=2.0)
yloc = plticker.MultipleLocator(base=.02)

ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.xaxis.set_major_locator(xloc)
ax.yaxis.set_major_locator(yloc)

# I want minor ticks for x axis
minor_xticks = np.linspace(0, 8, 8*5+1, endpoint=True)

# I want minor ticks for y axis
minor_yticks = np.linspace(.96-2*dy, 1, 5*2+1+2, endpoint=True)

ax.xaxis.set_ticks(minor_xticks, minor = True)
ax.yaxis.set_ticks(minor_yticks, minor = True)

# Specify different settings for major and minor grids
ax.grid(which = 'minor', alpha = 0.3)
ax.grid(which = 'major', alpha = 0.7)

# im = np.asarray(Image.open('/Users/alex/Desktop/img.png'))
# store.store(im, __file__, '_img')
im = store.load('results/data/numpy_objs/misc_acceptance_hmc1dfree_img.json')
ax.imshow(im, extent=xlim+ylim, aspect='auto', label = r'paper')

x  = np.linspace(0, 8, 101)
f  = np.vectorize(lambda t: prob(t, .2, 0, np.arange(1, 20+1)))

plt.plot(x, f(.5*x), linewidth=4., color='red', alpha=0.6, label = r'$(x,f(x)) \to (x, f(x/2))$')
ax.legend(loc='best', shadow=True, fancybox=True)
plt.draw()