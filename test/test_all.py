import numpy as np
rng = np.random.RandomState(1234)

import utils
import hmc

import test_potentials
import test_dynamics
import test_hmc
import test_lattice

def testPotentials():
    utils.newTest('potentials')
    test = test_potentials.Test()
    assert test.bivariateGaussian(save = False)
    assert test.qHO()

def testDynamics():
    utils.newTest('dynamics')
    
    integrator = hmc.dynamics.Leap_Frog(duE = None, n_steps = 100, step_size = 0.1) # grad set in test
    tests = test_dynamics.Test(dynamics = integrator)
    
    assert tests.constantEnergy(tol = 0.05,
        step_sample = np.linspace(1, 100, 10, True, dtype=int),
        step_sizes = np.linspace(0.01, 0.1, 5, True),
        save = False, print_out = True)
    
    assert tests.reversibility(steps = 1000, tol = 0.01, 
        save = False, print_out = True)


def testLattice():
    utils.newTest('lattice')
    test = test_lattice.Test()
    assert test.Wrap(print_out = True)
    assert test.Laplacian(print_out = True)

def testHMC():
    utils.newTest('hmc')
    m = hmc.Momentum(rng)
    test = test_hmc.Test(rng)
    
    assert m.test(print_out=False)
    assert test.hmcSho1d(n_samples = 100, n_burn_in = 1000,
        tol = 5e-2, print_out = True, save = False)[0]
    assert test.hmcGaus2d(n_samples = 10000, n_burn_in = 50,
        tol = 5e-2, print_out = True, save = False)[0]
if __name__ == '__main__':
    testPotentials()
    testDynamics()
    testLattice()
    testHMC()
    pass
