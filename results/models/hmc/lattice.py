from hmc.lattice import Periodic_Lattice
from hmc.hmc import *

class Model():
    """A model to sample for lattice QFT
    
    Required Inputs
        pot         :: potential class - see hmc.potentials
    
    Optional Inputs
        n_steps     :: int  :: default number of steps for dynamics
        step_size   :: int  :: default step size for dynamics
        n           :: int  :: lattice sites in each dimension
        spacing     :: float :: lattice spacing
    """
    def __init__(self, pot, n_steps=20, step_size=0.1, n=100, spacing=1.0):
        
        dim = 1
        self.n = n
        self.x0 = np.random.random((n,)*dim)
        self.x0 = Periodic_Lattice(array=self.x0, spacing=spacing)
        
        self.rng = np.random.RandomState(111)
        
        self.pot = pot
        self.dynamics = Leap_Frog(
            duE = self.pot.duE,
            step_size = step_size,
            n_steps = n_steps,
            lattice=True)
        
        self.sampler = Hybrid_Monte_Carlo(self.x0, self.dynamics, self.pot, self.rng)
    
    def run(self, n_samples, n_burn_in):
        """Runs the HMC algorithm to sample the potential
        
        Required Inputs
            n_samples   :: int  :: number of samples
            n_burn_in   :: int  :: number of burnin steps
        
        """
        p_samples, samples = self.sampler.sample(
            n_samples = n_samples, n_burn_in = n_burn_in)
        burn_in, samples = samples # return the shape: (n, dim, 1)
        
        # flatten last dimension to a shape of (n, dim)
        # this line doesn't work!
        # dim = self.x0.get.shape[0] # dimension the column vector
        dim = 1
        burn_in = np.asarray([i.get for i in burn_in]).reshape(n_burn_in+1, self.n)
        samples = np.asarray([i.get for i in samples]).reshape(n_samples+1, self.n)
        
        self.samples = samples
        self.burn_in = burn_in
        pass