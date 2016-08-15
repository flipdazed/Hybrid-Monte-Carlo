#!/usr/bin/env python
import numpy as np
from results.data.store import load
from correlations.acorr import acorr as getAcorr
from correlations.errors import uWerr, gW
from common.uWerrWind import plot, preparePlot
from common.utils import saveOrDisplay

save = False
file_name = 'uWerrWind_mag2_hmc.py'

dest = 'results/data/other_objs/uWerrWind_mag2_hmc_allPlot.pkl'
a = load(dest)


m         = 1.0
n_steps   = 10
step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps

n, dim = 100, 1
x0 = np.random.random((n,)*dim)
spacing = 0.05

op_name = r'$\hat{O} = \phi_0^2 :\phi_0 = \mathcal{F}^{-1}\tilde{\phi}_0 = '\
    + r' \sum_{x\in\mathbb{Z}^d_\text{L}}\tilde{\phi}_0$'
subtitle = r"Potential: {}; Lattice Shape: ".format('Klein Gordon') \
    + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        x0.shape, spacing, step_size, n_steps)

plot(save = saveOrDisplay(save, file_name), **a)