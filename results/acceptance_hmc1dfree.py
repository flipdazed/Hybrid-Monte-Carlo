#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt

from hmc.potentials import Klein_Gordon as KG
from common import acceptance

pot = KG(m=0.)
file_name = __file__

n, dim = 20, 1
x0 = np.random.random((n,)*dim)

if __name__ == '__main__':
    n_rng = np.arange(1, 41, 1)
    acceptance.main(x0, pot, file_name, n_rng, 
      n_samples = 1000000, step_size = .2, save = False)
