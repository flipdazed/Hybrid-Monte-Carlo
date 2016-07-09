#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt

from common import acceptance

file_name = __file__

n, dim = 20, 1
x0 = np.random.random((n,)*dim)

step_size = .2

if __name__ == '__main__':
    lengths = np.arange(1, 40, 1)
    acceptance.main(x0, file_name, lengths, save = True)