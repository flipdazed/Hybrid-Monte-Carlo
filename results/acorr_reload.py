#!/usr/bin/env python
import numpy as np
from results.data.store import load
from common.acorr import plot
from common.utils import saveOrDisplay

save = True
file_name = 'acorr_xx_hmc_kg'

dest = 'results/data/other_objs/{}_allPlot.pkl'.format(file_name)
a = load(dest)

plot(save = saveOrDisplay(save, file_name), **a)