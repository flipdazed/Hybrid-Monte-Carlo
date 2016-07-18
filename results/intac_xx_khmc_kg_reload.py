import numpy as np
from results.data.store import load
from common.intac import plot
from common.utils import saveOrDisplay

save = True
file_name = 'intac_xx_khmc_kg.py'

dest = 'results/data/other_objs/intac_xx_khmc_kg_allPlot.pkl'
a = load(dest)

plot(save = saveOrDisplay(save, file_name), **a)