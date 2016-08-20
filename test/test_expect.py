# -*- coding: utf-8 -*- 
import numpy as np
from scipy.special import erfc

import utils

from hmc.lattice import Periodic_Lattice
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO
from hmc.potentials import Klein_Gordon as KG
from hmc.hmc import *
from correlations import corr
import theory.operators
from models import Basic_HMC as Model

class Test(object):
    """Runs tests for expectation values
    
    Required Inputs
        rng :: np.random.RandomState :: must be able to call rng.uniform
    
    Optional Inputs
        length :: int :: 1d lattice length
        dim :: int  :: number of dimensions
        spacing :: float :: lattice spacing
    """
    def __init__(self, rng, spacing=.1, length = 101, dim = 1, verbose = False):
        self.id  = 'Expectations: <x(0)x(0)>, <exp{-𝛿H}>, <P_acc>'
        self.rng = rng
        self.length = length
        self.dim = dim
        self.spacing = spacing
        self.lattice_shape = (self.length,)*self.dim
        
        self.n = dim*length
    
    def kgDeltaH(self, n_steps = 20, step_size = .1, tol = 1e-1, print_out = True):
        """Tests to see if <exp{-\delta H}> == 1
        
        Optional Inputs
            n_steps         :: int      :: LF trajectory lengths
            step_size       :: int      :: Leap Frog step size
            tol :: float :: tolerance level of deviation from expected value
            print_out :: bool :: prints info if True
        """
        passed = True
        pot = KG()
        delta_hs, av_acc = self._runDeltaH(pot)
        meas_av_exp_dh  = np.asscalar(np.exp(-delta_hs).mean())
        passed *= np.abs(meas_av_exp_dh - 1) <= tol
        
        if print_out:
            utils.display('<exp{-𝛿H}> using ' + pot.name, passed,
                details = {
                    'Inputs':['a: {}'.format(self.spacing),'n steps: {}'.format(n_steps),
                        'step size: {}'.format(step_size),
                        'lattice: {}'.format(self.lattice_shape)],
                    'Outputs':['<exp{{-𝛿H}}>: {}'.format(meas_av_exp_dh),
                        '<Prob. Accept>: {}'.format(av_acc)]
                    })
        return passed
    
    def kgCorrelation(self, mu = 1., tol = 1e-2, print_out = True):
        """calculates the value <x(0)x(0)> for the QHO
        
        Optional Inputs
            mu :: float :: parameter used in potentials
            tol :: float :: tolerance level of deviation from expected value
            print_out :: bool :: prints info if True
        """
        passed = True
        
        pot = KG(m=mu)
        measured_xx = self._runCorrelation(pot)
        expected_xx = theory.operators.x2_1df(1, self.n, self.spacing, 0)
        
        passed *= np.abs(measured_xx - expected_xx) <= tol
        
        if print_out:
            utils.display('<x(0)x(t)> using ' + pot.name, passed,
                details = {
                    'Inputs':['a: {}'.format(self.spacing),'µ: {}'.format(mu),
                        'shape: {}'.format(self.lattice_shape), 'samples: {}'.format(self.n)],
                    'Outputs':['expected: {}'.format(expected_xx),
                        'measured: {}'.format(measured_xx)]
                    })
        return passed
    
    def kgAcceptance(self, n_steps = 20, step_size = .1, tol = 1e-2, print_out = True):
        """calculates the value <P_acc> for the free field
        
        Optional Inputs
            n_steps         :: int      :: LF trajectory lengths
            step_size       :: int      :: Leap Frog step size
            m :: float :: parameter used in potentials
            tol :: float :: tolerance level of deviation from expected value
            print_out :: bool :: prints info if True
        """
        passed = True
        
        pot = KG()
        n_samples = 1000
        delta_hs, measured_acc = self._runDeltaH(pot, n_samples = n_samples, 
            step_size=step_size, n_steps=n_steps)
        av_dh = np.asscalar(delta_hs.mean())
        expected_acc = erfc(.5*np.sqrt(av_dh))
        
        passed *= np.abs(measured_acc - expected_acc) <= tol
        
        meas_av_exp_dh  = np.asscalar(np.exp(-delta_hs).mean())
        if print_out:
            utils.display('<P_acc> using ' + pot.name, passed,
                details = {
                    'Inputs':['a: {}'.format(self.spacing),'m: {}'.format(pot.m),
                        'shape: {}'.format(self.lattice_shape), 'samples: {}'.format(n_samples),
                        'n steps: {}'.format(n_steps), 'step size: {}'.format(step_size)],
                    'Outputs':['expected: {}'.format(expected_acc),
                        'measured: {}'.format(measured_acc),
                        '<exp{{-𝛿H}}>: {}'.format(meas_av_exp_dh),
                        '<𝛿H>: {}'.format(av_dh)]
                    })
        return passed
    
    def _runDeltaH(self, pot, n_samples = 10000, n_burn_in = 20, n_steps=20, step_size=.1):
        """Obtains the average exponentiated change in H for 
        a given potential
        
        Optional Inputs
            n_samples   :: int  :: number of samples
            n_burn_in   :: int  :: number of burnin steps
            n_steps         :: int      :: LF trajectory lengths
            step_size       :: int      :: Leap Frog step size
        """
        x0 = np.random.random(self.lattice_shape)
        
        model = Model(x0, pot, spacing=self.spacing, n_steps=n_steps, step_size=step_size,      
                        accept_kwargs={'get_delta_hs':True})
        model.run(n_samples=n_samples, n_burn_in=n_burn_in, verbose = True)
        delta_hs = np.asarray(model.sampler.accept.delta_hs[n_burn_in:]).flatten()
        accept_rates = np.asarray(model.sampler.accept.accept_rates[n_burn_in:]).flatten()
        
        av_acc          = np.asscalar(accept_rates.mean())
        return (delta_hs, av_acc)
    
    def _runCorrelation(self, pot, n_samples = 10000, n_burn_in = 20):
        """Runs the correlation function calculation
        and instantiates all the necessary functions
        using an arbitrary potential
        
        Optional Inputs
            n_samples   :: int  :: number of samples
            n_burn_in   :: int  :: number of burnin steps
        """
        x0 = np.random.random(self.lattice_shape)
        model = Model(x0, pot, spacing=self.spacing)
        self.c = corr.Correlations_1d(model, attr_run='run', attr_samples='samples')
        self.c.runModel(n_samples = n_samples, n_burn_in = n_burn_in, verbose = True)
        xx = self.c.getTwoPoint(separation=0)
        return xx
#
if __name__ == '__main__':
    rng = np.random.RandomState(1241)
    utils.logs.logging.root.setLevel(utils.logs.logging.DEBUG)
    test = Test(rng)
    utils.newTest(test.id)
    test.kgAcceptance(1, .1)
    test.kgCorrelation()
    test.kgDeltaH()