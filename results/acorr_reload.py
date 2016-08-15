#!/usr/bin/env python
import numpy as np
from results.data.store import load
# from theory.autocorrelations import M2_Exp as Theory
from common.acorr import plot
from common.utils import saveOrDisplay

save = False
file_name = 'acorrIssues_mag2_hmc_lowM'

dest = 'results/data/other_objs/{}_allPlot.pkl'.format(file_name)
a = load(dest)
pa =  load('results/data/other_objs/{}_probs.pkl'.format(file_name))

# m         = 0.1
# n_steps   = 1000
# step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
# tau       = step_size*n_steps
#
# n, dim  = 10, 1
# x0 = np.random.random((n,)*dim)
# spacing = 1.0
#
# n_samples, n_burn_in = 1000000, 50
# c_len   = 100000
#
# th = Theory(tau=tau, m=m)
# vFn = lambda x: th.eval(t=x, pa=pa, theta=np.pi/2)/th.eval(t=0, pa=pa, theta=np.pi/2.)
#
# l = a['lines'].keys()[0]
# x, f0 = a['lines'][l]
# f1 = vFn(x)
# a['lines'][l] = (x, f1)

n = 450
k = a['acns'].keys()[0]
x,y, e = a['acns'][k]
a['acns'][k] = x[:n], y[:n], e[:n]

# from scipy.optimize import curve_fit
# fn = lambda x, a, b, c: a*np.cos(b*x)**2+c
# fit,cov = curve_fit(fn, x[:100], y[:100])
#
# l = a['lines'].keys()[0]
# x, f0 = a['lines'][l]
# f1 = fn(x, *fit)
# a['lines'][l] = (x, f1)

plot(save = saveOrDisplay(save, file_name), **a)
