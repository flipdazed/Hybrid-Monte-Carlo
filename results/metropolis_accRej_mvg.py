#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import numpy as np

from common.hmc_model import Model
from hmc.potentials import Multivariate_Gaussian as MVG

import metropolis_qho_accRej

#
if __name__ == '__main__':
    
    n_burn_in = 15
    n_samples = 100
    
    pot = MVG()
    
    x0 = np.asarray([[-4.],[4.]])
    model = Model(x0, pot=pot)
    model.sampler.accept.store_acceptance = True
    
    print 'Running Model: {}'.format(__file__)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    print 'Finished Running Model: {}'.format(__file__)
    
    accept_rates = np.asarray(model.sampler.accept.accept_rates).ravel()
    accept_rejects = np.asarray(model.sampler.accept.accept_rejects).ravel()
    delta_hs = np.asarray(model.sampler.accept.delta_hs).ravel()
    
    print '\n\t<Prob. Accept>: {:4.2f}'.format(accept_rates.mean(axis=0))
    print '\t<Prob. Accept>: {:4.2f}     (Measured)'.format(accept_rejects.mean(axis=0))
    print '\t<exp{{-𝛿H}}>:     {:8.2E} (Measured)\n'.format(delta_hs.mean(axis=0))
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    
    metropolis_qho_accRej.plot(probs=accept_rates,
        save = save_name
        # save = False
        )