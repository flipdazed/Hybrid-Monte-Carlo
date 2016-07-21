#!/usr/bin/env python
import numpy as np
from results.data.store import load
from common.acorr import plot
from common.utils import saveOrDisplay

save = True
file_name = 'acorr_xx_khmc_kg'

dest = 'results/data/other_objs/{}_allPlot.pkl'.format(file_name)
a = load(dest)

op_name = r'$\hat{O} = \sum_{pq} \Omega \phi_p\phi_q :\Omega = \delta_{p0}\delta{q0}$'

a['op_name'] = op_name

plot(save = saveOrDisplay(save, file_name), **a)