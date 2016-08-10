#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
from models import Basic_HMC as Model
from hmc.potentials import Klein_Gordon as KG

pot = KG(m=0.)

n, dim = 20, 1
x0 = np.random.random((n,)*dim)

n_samples = 10000
n_burn_in = 50
step_size = .2
n_steps   = 20

model = Model(x0.copy(), pot, step_size=step_size, n_steps=n_steps)
model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose=False)