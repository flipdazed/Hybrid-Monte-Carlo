from copy import copy
import numpy as np
rng = np.random.RandomState(1234)

import utils
import hmc
from hmc.potentials import Simple_Harmonic_Oscillator, Klein_Gordon
from hmc.dynamics import Leap_Frog
from hmc.lattice import Periodic_Lattice

import test_potentials
import test_dynamics
import test_hmc
import test_lattice
import test_momentum

import logging
logging.root.setLevel(logging.DEBUG)

def testPotentials():
    test = test_potentials.Test()
    utils.newTest(test.id)
    assert test.bvg()
    assert test.qho()
    pass

def testContinuumDynamics():
    utils.newTest('hmc.dynamics (continuum)')
    
    n_steps     = 100
    step_size   = .1
    tol         = .05
    
    pot = Simple_Harmonic_Oscillator()
    integrator = Leap_Frog(duE = None,
        n_steps = n_steps, step_size = step_size) # grad set in test
    tests = test_dynamics.Continuum(dynamics = integrator, pot = pot)
    
    p, x = np.asarray([[4.]]), np.asarray([[1.]])
    
    assert tests.constantEnergy(p, x, tol = tol,
        step_sample = np.linspace(1, n_steps, 10, True, dtype=int),
        step_sizes = np.linspace(0.01, step_size, 5, True),
        save = False, print_out = True)
    
    assert tests.reversibility(p, x, steps = n_steps*10, tol = tol, 
        save = False, print_out = True)
    pass

def testLatticeDynamics():
    utils.newTest('hmc.dynamics  (lattice)')
    
    n           = 10
    spacing     = 1.
    n_steps     = 500
    step_size   = .01
    dim         = 1
    tol         = .05
    
    pot = Klein_Gordon()
    integrator = Leap_Frog(lattice=True, duE = None,    # grad set in test
        n_steps = n_steps, step_size = step_size)
    tests = test_dynamics.Lattice(dynamics = integrator, pot = pot)
    
    x_nd = np.random.random((n,)*dim)
    p0 = np.random.random((n,)*dim)
    x0 = Periodic_Lattice(array=x_nd, spacing=spacing)
    
    assert tests.constantEnergy(p0, x0, tol = tol,
        step_sample = [n_steps],
        step_sizes = [step_size],
        save = False, print_out = True)
    
    assert tests.reversibility(p0, x0, steps = n_steps, tol = tol, 
        save = False, print_out = True)
    
    pass

def testLattice():
    utils.newTest('lattice')
    test = test_lattice.Test()
    assert test.wrap(print_out = True)
    assert test.laplacian(print_out = True)
    assert test.gradSquared(symmetric = False, print_out = True)
    assert test.gradSquared(symmetric = True, print_out = True)
    pass

def testHMC():
    utils.newTest('hmc')
    test = test_hmc.Test(rng)
    
    n_samples   = 1000
    n_burn_in   = 50
    tol         = 5e-1
    
    assert test.hmcSho1d(n_samples = n_samples, n_burn_in = n_burn_in,
        tol = tol, print_out = True, save = False)[0]
    assert test.hmcGaus2d(n_samples = n_samples, n_burn_in = n_burn_in,
        tol = tol, print_out = True, save = False)[0]
    pass

def testMomentum():
    utils.newTest('hmc.Momentum')
    test = test_momentum.Test(rng=rng)

    rand4 = np.random.random(4)
    p41 = np.mat(rand4.reshape(4,1))
    p22 = np.mat(rand4.reshape(2,2))
    p14 = np.mat(rand4.reshape(1,4))
    
    assert test.vectors(p41, print_out = True)
    assert test.vectors(p22, print_out = True)
    assert test.vectors(p14, print_out = True)
    
    assert test.mixing(print_out = True)
    pass

if __name__ == '__main__':
    testPotentials()
    testContinuumDynamics()
    testLatticeDynamics()
    testLattice()
    testHMC()
    testMomentum()
    pass