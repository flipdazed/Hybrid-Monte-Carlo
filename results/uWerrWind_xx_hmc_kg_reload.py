#!/usr/bin/env python
import numpy as np
from results.data.store import load
from correlations.acorr import acorr as getAcorr
from correlations.errors import uWerr, gW
from common.errs import plot, preparePlot
from common.utils import saveOrDisplay

save = True
file_name = 'errs_xx_kg.py'

dest = 'results/data/numpy_objs/errs_xx_kg_cfn.json'
a = load(dest)

op_name = r'$\langle x^2 \rangle_{\text{L}}$'
subtitle = r"Potential: {}; Lattice Shape: ".format('Klein Gordon') \
    + r"${}$; $a={:.1f}; \delta\tau={:.1f}; n={}$".format(
        '(20,)', 1., .1, 20)

ans   = uWerr(a)
lines, labels, w = preparePlot(a, ans, n=a.shape[0])

plot(lines, w, subtitle,
    labels = labels, op_name = op_name,
    save = saveOrDisplay(save, file_name))