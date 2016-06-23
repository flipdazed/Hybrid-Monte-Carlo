import numpy as np
import matplotlib.pyplot as plt
import subprocess

import utils

from hmc import checks
from hmc.lattice import Periodic_Lattice
from hmc.potentials import Multivariate_Gaussian as MVG
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

class Test(object):
    def __init__(self, print_out=True):
        self.id = 'initialise potentials'
        self.print_out = print_out
        self.fns = { 'potentialEnergy':'uE',
                'gradPotentialEnergy':'duE',
                'kineticEnergy':'kE'}
        pass
    
    def bvg(self):
        """Plots a test image of the Bivariate Gaussian"""
        passed = True
        
        mean = np.asarray([[0.], [0.]])
        cov = np.asarray([[1.0,0.8],[0.8,1.0]])
        
        self.pot = MVG(mean = mean, cov = cov)
        self.x = np.asarray([[-3.5], [4.]])
        self.p = np.asarray([[ 1.],  [2.]])
        idx_list = ['one_iteration_garbage']
        
        passed = self._TestFns("BVG Potential", passed, self.x, self.p)
        return passed
    
    def qho(self, dim = 4, sites = 10, spacing = 1.):
        """checks that QHO can be initialised and all functions run"""
        
        passed = True
        shape = (sites,)*dim
        raw_lattice = np.arange(sites**dim).reshape(shape)
        
        self.pot = QHO()
        self.x = Periodic_Lattice(array = raw_lattice, spacing = 1.)
        self.p = np.asarray(shape)
        idx_list = [(0,)*dim, (sites,)*dim, (sites-1,)*dim]
        
        passed = self._TestFns("QHO Potential", passed, self.x, self.p, idx_list)
        
        return passed
    
    def _TestFns(self, name, passed, x, p, idx_list=[0]):
        """Returns a list of functions for the current potential
        
        Required Inputs:
            name     :: string           :: name of the test
            passed   :: bool             :: current pass state
            idx_list :: tuple (np shape) :: test indices for lattice gradients
        
        Expectations:
            self.pot :: contains the potential
        
        Return value is:
            [(full name function_i, abbbrev. name function_i), ... ]
        """
        
        assert hasattr(self, 'pot')
        
        # get a list of the functions from the potential
        f_list = [getattr(self.pot, abbrev) for full,abbrev in self.fns.iteritems()]
        
        # create a list of passed functions
        passed_fns = []
        failed_fns = []
        for i in f_list: # iterate functions
            for idx in idx_list:                # iterate through indices
                try:
                    i_ret = i(x=x, p=p, idx=idx)
                    passed_fns.append(i.__name__+':{}'.format(idx))
                except Exception as e:
                    passed = False
                    checks.fullTrace()
                    failed_fns.append(i.__name__+':{} :: {}'.format(idx, e))
        
        if self.print_out:
            utils.display(name, passed,
                details = {
                    'passed functions':passed_fns,
                    'failed functions':failed_fns
                    })
        
        return passed
#
if __name__ == '__main__':
    test = Test()
    utils.newTest(test.id)
    test.bvg()
    test.qho()