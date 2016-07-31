#!/usr/bin/env python
import numpy as np
from results.data.store import load
from common.acorr import plot
from common.utils import saveOrDisplay
from theory.autocorrelations import M2_Exp as Theory

save = False
file_name = 'acorr_xx_hmc_kg'

dest = 'results/data/other_objs/{}_allPlot.pkl'.format(file_name)
a = load(dest)

m = 0.4
step_size = 1/((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/20.
n_steps   = 20

tau = (n_steps*step_size)
th = Theory(tau=tau, m=m)
vFn = lambda x: th.eval(t=x, pa=0.65, theta=np.pi/2)/th.eval(t=0, pa=0.65, theta=np.pi/2)
        
l = a['lines'].keys()[0]
x, f0 = a['lines'][l]
f1 = vFn(x)
a['lines'][l] = (x, f1)

plot(save = saveOrDisplay(save, file_name), **a)