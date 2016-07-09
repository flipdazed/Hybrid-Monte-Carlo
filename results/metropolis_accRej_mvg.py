#!/usr/bin/env python
import numpy as np

from common import metropolis_accRej
from hmc.potentials import Multivariate_Gaussian as MVG

file_name = __file__
pot = MVG()


x0 = np.asarray([[-4.],[4.]])

n_burn_in, n_samples = 15, 100

if __name__ == '__main__':
    metropolis_accRej.main(x0, pot, file_name, n_samples, n_burn_in,
        save = False)