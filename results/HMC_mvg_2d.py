from hmc.potentials import Multivariate_Gaussian
from hmc.hmc import *

class Model():
    """A model to sample the MVG
    
    Optional Inputs
        n_steps     :: int  :: default number of steps for dynamics
        step_size   :: int  :: default step size for dynamics
    """
    def __init__(self, n_steps=10, step_size=0.1):
        
        self.x0 = np.asarray([[-3.5], [4.]])
        
        self.rng = np.random.RandomState(111)
        
        self.pot = Multivariate_Gaussian()
        self.dynamics = Leap_Frog(
            duE = self.pot.duE,
            step_size = step_size,
            n_steps = n_steps)
        
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
        dim = self.x0.shape[0] # dimension the column vector
        self.samples = np.asarray(samples).T.reshape(dim, -1).T
        self.burn_in = np.asarray(burn_in).T.reshape(dim, -1).T
        pass