#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt

from common import acceptance

file_name = __file__

n, dim = 20, 1
x0 = np.random.random((n,)*dim)

if __name__ == '__main__':
    lengths = np.arange(1, 41, 1)
    acceptance.main(x0, file_name, lengths, 
      n_samples = 10000, step_size = .2, save = True)
