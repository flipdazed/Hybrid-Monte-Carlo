#!/usr/bin/env python
import numpy as np

from common import dynamics_rev
from hmc.potentials import Klein_Gordon as KG

file_name = __file__
pot = KG()

n, dim = 100, 1
x0 = np.random.random((n,)*dim)



if '__main__' == __name__:
    dynamics_rev.main(x0, pot, file_name, save = True)