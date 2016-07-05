
import numpy as np
rng = np.random.RandomState(1234)

import utils
import hmc
from hmc.potentials import Simple_Harmonic_Oscillator
from hmc.potentials import Quantum_Harmonic_Oscillator
from hmc.potentials import Klein_Gordon
from hmc.dynamics import Leap_Frog
from hmc.lattice import Periodic_Lattice

import test_potentials
import test_dynamics
import test_hmc
import test_lattice
import test_momentum
import test_expect

import logging
logging.root.setLevel(logging.DEBUG)

def testExpectations():
    test = test_expect.Test(rng=rng, spacing=.1, length = 100, dim = 1)
    utils.newTest(test.id)
    assert test.qhoCorellation(mu = 1., tol = 1e-2)
    assert test.qhoDeltaH(tol = 1e-1)
    pass

def testPotentials():
    test = test_potentials.Test()
    utils.newTest(test.id)
    assert test.bvg()
    assert test.qho()
    pass


def testDynamics():
    dim         = 1
    n           = 10
    spacing     = 1.
    step_size   = [0.01, .1]
    n_steps     = [1, 500]
    
    samples     = 5
    step_sample = np.linspace(n_steps[0], n_steps[1], samples, True, dtype=int),
    step_sizes = np.linspace(step_size[0], step_size[1], samples, True)
    
    x_nd = np.random.random((n,)*dim)
    p0 = np.random.random((n,)*dim)
    x0 = Periodic_Lattice(x_nd)
    
    dynamics = Leap_Frog(
        duE = None,
        n_steps = n_steps[-1],
        step_size = step_size[-1])
    
    for pot in [Simple_Harmonic_Oscillator(), 
        Klein_Gordon(), Quantum_Harmonic_Oscillator()]:
        
        dynamics.duE = pot.duE
        
        test = test_dynamics.Constant_Energy(pot, dynamics)
        utils.newTest(test.id)
        assert test.run(p0, x0, step_sample, step_sizes)
        
        test = test_dynamics.Reversibility(pot, dynamics)
        utils.newTest(test.id)
        assert test.run(p0, x0)
    pass

def testLattice():
    test = test_lattice.Test()
    utils.newTest(test.id)
    assert test.wrap(print_out = True)
    assert test.laplacian(print_out = True)
    assert test.gradSquared(print_out = True)
    pass

def testHMC():
    test = test_hmc.Test(rng)
    utils.newTest(test.id)
    
    n_burn_in   = 15
    tol         = 1e-1
    
    assert test.hmcSho1d(n_samples = 1000, n_burn_in = n_burn_in, tol = tol)
    # assert test.hmcGaus2d(n_samples = 10000, n_burn_in = n_burn_in, tol = tol)
    assert test.hmcQho(n_samples = 100, n_burn_in = n_burn_in, tol = tol)
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
    testExpectations()
    testPotentials()
    testDynamics()
    testLattice()
    testHMC()
    testMomentum()
    pass