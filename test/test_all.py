import numpy as np
rng = np.random.RandomState(1234)

import utils
import hmc

import test_potentials
import test_dynamics
import test_hmc
import test_lattice
import test_momentum

def testPotentials():
    utils.newTest('potentials')
    test = test_potentials.Test()
    assert test.bivariateGaussian(save = False)
    assert test.lattice_qHO()
    pass

def testDynamics():
    utils.newTest('dynamics')
    
    pot = hmc.potentials.Simple_Harmonic_Oscillator(k = 1.)
    integrator = hmc.dynamics.Leap_Frog(duE = None, 
        n_steps = 100, step_size = 0.1) # grad set in test
    tests = test_dynamics.Test(dynamics = integrator, pot = pot)
    
    p, x = np.asarray([[4.]]), np.asarray([[1.]])
    assert tests.constantEnergy(p, x, tol = 0.05,
        step_sample = np.linspace(1, 100, 10, True, dtype=int),
        step_sizes = np.linspace(0.01, 0.1, 5, True),
        save = False, print_out = True)
    
    assert tests.reversibility(p, x, steps = 1000, tol = 0.01, 
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
    
    assert test.hmcSho1d(n_samples = 10000, n_burn_in = 50,
        tol = 5e-2, print_out = True, save = False)[0]
    assert test.hmcGaus2d(n_samples = 10000, n_burn_in = 50,
        tol = 5e-2, print_out = True, save = False)[0]
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
    testDynamics()
    testLattice()
    testHMC()
    testMomentum()
    pass
