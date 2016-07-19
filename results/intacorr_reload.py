#!/usr/bin/env python
import numpy as np
from results.data.store import load
from common.intac import plot
from common.utils import saveOrDisplay

save = True
file_name = 'intac_xx_ghmc_kg_0p00t0p50.py'

dest = 'results/data/other_objs/intac_xx_ghmc_kg_0p00t0p50_allPlot.pkl'
a = load(dest)

a['subtitle'] = r"Potential: {}; Lattice: ".format('Klein-Gordon') \
    + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        (20,), 1.0, 0.1, 20)

plot(save = saveOrDisplay(save, file_name), **a)