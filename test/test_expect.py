# -*- coding: utf-8 -*- 
import numpy as np

import utils

from hmc.lattice import Periodic_Lattice
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO
from hmc.hmc import *
from correlations import corr
from models import Basic_HMC as Model

class Test(object):
    """Runs tests for the expectation of <x(0)x(0)>
    
    Required Inputs
        rng :: np.random.RandomState :: must be able to call rng.uniform
    
    Optional Inputs
        length :: int :: 1d lattice length
        dim :: int  :: number of dimensions
        spacing :: float :: lattice spacing
    """
    def __init__(self, rng, spacing=.1, length = 100, dim = 1, verbose = False):
        self.id  = '<x(0)x(0)>'
        self.rng = rng
        self.length = length
        self.dim = dim
        self.spacing = spacing
        self.lattice_shape = (self.length,)*self.dim
        
        self.n = dim*length
        
    def qho(self, mu = 1., tol = 1e-2, print_out = True):
        """calculates the value <x(0)x(0)> for the QHO
        
        Optional Inputs
            mu :: float :: parameter used in potentials
            tol :: float :: tolerance level of deviation from expected value
            print_out :: bool :: prints info if True
        """
        passed = True
        
        pot = QHO(mu=mu)
        measured_xx = self._run(pot)
        expected_xx = corr.qho_theory(self.spacing, mu, self.n)
        passed = np.abs(measured_xx - expected_xx) <= tol
        
        if print_out:
            utils.display(pot.name, passed,
                details = {
                    'Inputs':['a: {}'.format(self.spacing),'µ: {}'.format(mu),
                        'shape: {}'.format(self.lattice_shape), 'n: {}'.format(self.n)],
                    'another output':['expected: {}'.format(expected_xx),
                        'measured: {}'.format(measured_xx)]
                    })
        return passed
    
    def _run(self, pot, n_samples = 100, n_burn_in = 20):
        """Runs the correlation function calculation
        and instantiates all the necessary functions
        using an arbitrary potential
        
        Optional Inputs
            n_samples   :: int  :: number of samples
            n_burn_in   :: int  :: number of burnin steps
        """
        x0 = np.random.random(self.lattice_shape)
        model = Model(x0, pot, spacing=self.spacing)
        self.c = corr.Corellations_1d(model, 'run', 'samples')
        self.c.runModel(n_samples = n_samples, n_burn_in = n_burn_in, verbose = True)
        xx = self.c.twoPoint(separation=0)
        return xx
#
if __name__ == '__main__':
    rng = np.random.RandomState(1241)
    utils.logs.logging.root.setLevel(utils.logs.logging.DEBUG)
    test = Test(rng)
    utils.newTest(test.id)
    test.qho()