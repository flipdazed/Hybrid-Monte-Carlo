#!/usr/bin/env python
import numpy as np
from results.data.store import load
from common.intac import plot
from common.utils import saveOrDisplay

save = True
file_name = 'intacorr_xx_ghmc_kg_0p00t2p00'

dest = 'results/data/other_objs/{}_allPlot.pkl'.format(file_name)
a = load(dest)

a['subtitle'] = r"Potential: {}; Lattice: ".format('Klein-Gordon') \
    + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        (20,), 1.0, 0.1, 20)

plot(save = saveOrDisplay(save, file_name), **a)