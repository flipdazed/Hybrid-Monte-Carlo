#!/usr/bin/env python
import numpy as np
from results.data.store import load
from common.errs import plot
from common.utils import saveOrDisplay

save = False
file_name = 'errs_xx_kg.py'

dest = 'results/data/other_objs/errs_xx_khmc_kg_allPlot.pkl'
a = load(dest)

plot(save = saveOrDisplay(save, file_name), **a)