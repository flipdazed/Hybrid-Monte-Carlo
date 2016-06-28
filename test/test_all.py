from copy import copy
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

import logging
logging.root.setLevel(logging.DEBUG)

def testPotentials():
    test = test_potentials.Test()
    utils.newTest(test.id)
    assert test.bvg()
    assert test.qho()
    pass

def testContinuumDynamics():
    step_size   = [0.01, .1]
    n_steps     = [1, 500]
    tol         = .05
    samples     = 5
    
    step_sample = np.linspace(n_steps[0], n_steps[1],
        samples, True, dtype=int),
    step_sizes = np.linspace(step_size[0], step_size[1],
        samples, True)
    
    pot = Simple_Harmonic_Oscillator()
    p0  = np.asarray([[4.]])
    x0  = np.asarray([[1.]])
    
    dynamics = Leap_Frog(
        duE = pot.duE,
        n_steps = n_steps[-1],
        step_size = step_size[-1],
        lattice = False)
    
    test = test_dynamics.Constant_Energy(pot, dynamics, tol=tol)
    utils.newTest(test.id)
    assert test.continuum(p0, x0, step_sample, step_sizes)
    
    test = test_dynamics.Reversibility(pot, dynamics, tol=tol)
    utils.newTest(test.id)
    assert test.continuum(p0, x0)
    
    pass

def testLatticeDynamics():
    dim         = 1
    n           = 10
    spacing     = 1.
    step_size   = [0.01, .1]
    n_steps     = [1, 500]
    
    samples     = 2
    tol         = .05
    
    x_nd = np.random.random((n,)*dim)
    p0 = np.random.random((n,)*dim)
    x0 = Periodic_Lattice(array=copy(x_nd), spacing=spacing)
    
    step_sample = np.linspace(n_steps[0], n_steps[1],
        samples, True, dtype=int),
    step_sizes = np.linspace(step_size[0], step_size[1],
        samples, True)
    
    def run(pot):
        """run with arbitrary potentials"""
    
        dynamics = Leap_Frog(
            duE = pot.duE,
            n_steps = n_steps[-1],
            step_size = step_size[-1],
            lattice = True)
    
        test = test_dynamics.Constant_Energy(pot, dynamics, tol=tol)
        utils.newTest(test.id)
        assert test.lattice(p0, x0, step_sample, step_sizes)
    
        test = test_dynamics.Reversibility(pot, dynamics, tol=tol)
        utils.newTest(test.id)
        assert test.lattice(p0, x0)
        pass
    
    run(Klein_Gordon())
    run(Quantum_Harmonic_Oscillator())
    pass

def testLattice():
    test = Test()
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
    testPotentials()
    testContinuumDynamics()
    testLatticeDynamics()
    testLattice()
    testHMC()
    testMomentum()
    pass