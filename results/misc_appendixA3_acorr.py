#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from PIL import Image
from scipy.signal import residuez, tf2zpk

from data import store
from plotter import Pretty_Plotter, PLOT_LOC
from common.utils import saveOrDisplay
from theory.autocorrelations import M2_Exp

file_name = __file__
save = False

# start plotting
pp = Pretty_Plotter()
pp._teXify() # LaTeX
pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
pp._updateRC()

dx = (0-2000)/10.
dy = (.0-.05)/5.

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

fig.suptitle(r"Verifying theory", fontsize=pp.tfont)
ax.set_title(r'Overlaying functions on an image of a published figure', fontsize=pp.tfont)
ax.set_xlabel(r'$t$', fontsize=pp.tfont)
ax.set_ylabel(r'$C_{M^2}(t)$', fontsize=pp.tfont)

xlim = (0, 2000)
ylim = (0, .05)
xloc = plticker.MultipleLocator(base=200)
yloc = plticker.MultipleLocator(base=.01)

ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.xaxis.set_major_locator(xloc)
ax.yaxis.set_major_locator(yloc)

# I want minor ticks for x axis
minor_xticks = np.linspace(0, 2000, 10*2+1, endpoint=True)

# I want minor ticks for y axis
minor_yticks = np.linspace(0, .05, 5*2+1, endpoint=True)

ax.xaxis.set_ticks(minor_xticks, minor = True)
ax.yaxis.set_ticks(minor_yticks, minor = True)

# Specify different settings for major and minor grids
ax.grid(which = 'minor', alpha = 0.3)
ax.grid(which = 'major', alpha = 0.7)

# im = np.asarray(Image.open('results/figures/archive/misc_appendixA3_acorr_img.png'))
# store.store(im, __file__, '_img')
im = store.load('results/data/numpy_objs/misc_appendixA3_acorr_img.json')
ax.imshow(im, extent=xlim+ylim, aspect='auto', label = r'paper')

m = 0.01
r = iter([np.sqrt(3)*m, .5*m*(3*np.sqrt(3)+np.sqrt(15)), .5*m*(3*np.sqrt(3)-np.sqrt(15))])

f = M2_Exp(next(r), m)
x = np.linspace(0, 1000, 1000, True)

plt.plot(x, f.eval(x), linewidth=4., color='red', alpha=0.6, linestyle="--", 
    label = r'$r = \sqrt{3}m$')
plt.plot(x, f.eval(x, tau=1/next(r)), linewidth=4., color='green', alpha=0.6, linestyle="--",
    label = r'$r = \frac{m}{2}(3\sqrt{3} + \sqrt{15})$')
plt.plot(x, f.eval(x, tau=1/next(r)), linewidth=4., color='blue', alpha=0.6, linestyle="--",
    label = r'$r = \frac{m}{2}(3\sqrt{3} - \sqrt{15})$')

ax.legend(loc='lower right', shadow=True, fancybox=True)

pp.save_or_show(saveOrDisplay(save, file_name), PLOT_LOC)