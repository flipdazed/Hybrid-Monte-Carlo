from hmc.lattice import Periodic_Lattice
from hmc.hmc import *

class Model():
    """A model to sample from the potentials with LeapFrog
    
    Required Inputs
        x0          :: position (lattice)
        pot         :: potential class - see hmc.potentials
    
    Optional Inputs
        n_steps     :: int  :: default number of steps for dynamics
        step_size   :: int  :: default step size for dynamics
        spacing     :: float :: lattice spacing
    """
    def __init__(self, x0, pot, n_steps=20, step_size=0.1, spacing=1.):
        
        self.x0 = Periodic_Lattice(x0, lattice_spacing=spacing)
        
        self.rng = np.random.RandomState(111)
        
        self.pot = pot
        self.dynamics = Leap_Frog(
            duE = self.pot.duE,
            step_size = step_size,
            n_steps = n_steps)
        
        self.sampler = Hybrid_Monte_Carlo(self.x0, self.dynamics, self.pot, self.rng)
    
    def run(self, n_samples, n_burn_in, verbose = False):
        """Runs the HMC algorithm to sample the potential
        
        Required Inputs
            n_samples   :: int  :: number of samples
            n_burn_in   :: int  :: number of burnin steps
        
        Optional Inputs
            verbose :: bool :: a progress bar if True
        """
        p_samples, samples = self.sampler.sample(
            n_samples = n_samples, n_burn_in = n_burn_in, verbose = verbose)
        burn_in, samples = samples # return the shape: (n, dim, 1)
        
        # flatten last dimension to a shape of (n, dim)
        self.samples = np.asarray(samples).reshape(n_samples+1, -1)
        self.burn_in = np.asarray(burn_in).reshape(n_burn_in+1, -1)
        pass