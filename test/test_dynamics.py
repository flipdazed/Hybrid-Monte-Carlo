import numpy as np
import matplotlib.pyplot as plt

import utils
from hmc import checks

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.dynamics import Leap_Frog

from hmc.potentials import Simple_Harmonic_Oscillator as SHO
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO
from hmc.potentials import Klein_Gordon as KG

from hmc.lattice import Periodic_Lattice

class Constant_Energy(object):
    """Checks that the change in hamiltonian ~0
        for varying step_sizes and step_lengths
    
    Required Inputs
        pot         :: potential :: see hmc.potentials
        dynamics    :: dynamics :: see hmc.dynamics.Leap_Frog()
    
    Optional Inputs
        tol         :: float    :: tolerance level for hamiltonian changes
        print_out   :: bool     :: if True prints to screen
    """
    def __init__(self, pot, dynamics, tol = 1e-1, print_out = True):
        self.id         = 'Dynamics - Constant Energy :: {}'.format(pot.name)
        self.dynamics   = dynamics
        self.pot        = pot
        self.tol        = tol
        self.print_out  = print_out
    
    def run(self, p0, x0, step_sample, step_sizes):
        """ same for both lattice and non lattice
        
        Required Inputs
            p0          :: lattice    :: momentum
            x0          :: lattice    :: position
            step_sample :: np.array   :: array of steps lengths to test
            step_sizes  :: np.array   :: array of step sizes to test
        
        Returns
            passed :: bool :: True if passed
        """
        
        passed = True
        
        # calculate original hamiltonian and set starting vals
        h_old = self.pot.hamiltonian(p0, x0)
        
        # initial vals required to print out values associated
        # with the worst absolute deviation from perfect energy conservation
        # (0 = no energy loss)
        w_bmk  = 1.  # worst benchmark value
        diff   = 0.  # absolute difference
        w_step = 0   # worst steps
        w_size = 0   # worst step size
        
        # set up a mesh grid of the steps and sizes
        step_sample, step_sizes = np.meshgrid(step_sample, step_sizes)
        
        for n_steps_i, step_size_i in zip(np.ravel(step_sample), np.ravel(step_sizes)):
            
            # set new parameters
            self.dynamics.n_steps = n_steps_i
            self.dynamics.step_size = step_size_i
            
            # obtain new duynamics and resultant hamiltonian
            self.dynamics.newPaths()
            pf, xf = self.dynamics.integrate(p0.copy(), x0.copy())
            h_new = self.pot.hamiltonian(pf, xf)
            
            bench_mark = np.exp(-(h_old-h_new))
            new_diff = np.abs(1. - bench_mark)
            
            if self.print_out:# avoid calc every time when no print out
                if new_diff > np.abs(1. - w_bmk): # compare to last diff
                    w_bmk = bench_mark
                    w_h_new = h_new
                    w_step = n_steps_i
                    w_size = step_size_i
            
            # set new diff
            diff =  new_diff
            
            passed *= (diff <= self.tol).all()
            
        if self.print_out:
            utils.display(test_name='Constant Energy', outcome=passed,
                details = {
                    'initial H(p, x): {}'.format(h_old):[],
                    'worst   H(p, x): {}'.format(w_h_new):[
                            'steps: {}'.format(w_step),
                            'step size: {}'.format(w_size)],
                    'np.abs(1-exp(-dH)): {}'.format(np.abs(1. - w_bmk)):[]
                })
        
        return passed
    
#
class Reversibility(object):
    """Checks the integrator is reversible
        by running and then reversing the integration and 
        verifying the same point in phase space
    
    Required Inputs
        pot         :: potential :: see hmc.potentials
        dynamics    :: dynamics :: see hmc.dynamics.Leap_Frog()
    
    Optional Inputs
        tol         :: float    :: tolerance level for hamiltonian changes
        print_out   :: bool     :: if True prints to screen
    """
    def __init__(self, pot, dynamics, tol = 1e-1, print_out = True):
        self.id         = 'Dynamics - Reversibility :: {}'.format(pot.name)
        self.dynamics   = dynamics
        self.pot        = pot
        self.tol        = tol
        self.print_out  = print_out
        pass
    
    def run(self, p0, x0):
        """Checks the integrator is reversible
        by running and then reversing the integration and 
        verifying the same point in phase space
        
        Required Inputs
            p0          :: lattice :: momentum
            x0          :: lattice :: position
        """
        passed = True
        
        self.dynamics.newPaths()
        pm, xm = self.dynamics.integrate(p0.copy(), x0.copy())
        p0f, x0f = self.dynamics.integrate(-pm, xm) # time flip
        p0f = -p0f # time flip to point in right time again
        
        phase_change = np.linalg.norm( # calculate frobenius norm
            np.asarray([[p0f], [x0f]]) - np.asarray([[p0], [x0]])
            )
        passed = (phase_change < self.tol)
        
        if self.print_out: 
            utils.display(test_name="Reversibility of Integrator", 
            outcome=passed,
            details={
                'initial (p, x): ({}, {})'.format(p0, x0):[],
                'middle  (p, x): ({}, {})'.format(pm, xm):[],
                'final   (p, x): ({}, {})'.format(p0f, x0f):[],
                'phase change:    {}'.format(phase_change):[],
                'number of steps: {}'.format(self.dynamics.n_steps):[]
                })
        
        return passed
#
if __name__ == '__main__':
    
    # utils.logs.logging.root.setLevel(utils.logs.logging.DEBUG)
    
    dim         = 1
    n           = 10
    spacing     = 1.
    step_size   = [0.01, .1]
    n_steps     = [1, 500]
    
    samples     = 5
    step_sample = np.linspace(n_steps[0], n_steps[1], samples, True, dtype=int),
    step_sizes = np.linspace(step_size[0], step_size[1], samples, True)
    
    for pot in [SHO(), KG(), QHO()]:
        
        x_nd = np.random.random((n,)*dim)
        p0 = np.random.random((n,)*dim)
        x0 = Periodic_Lattice(x_nd)
        
        dynamics = Leap_Frog(
            duE = pot.duE,
            n_steps = n_steps[-1],
            step_size = step_size[-1])
        
        test = Constant_Energy(pot, dynamics)
        utils.newTest(test.id)
        test.run(p0, x0, step_sample, step_sizes)
        
        test = Reversibility(pot, dynamics)
        utils.newTest(test.id)
        test.run(p0, x0)