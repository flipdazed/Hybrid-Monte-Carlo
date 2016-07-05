import numpy as np
from tqdm import tqdm

from . import checks
from dynamics import Leap_Frog
from metropolis import Accept_Reject

class Hybrid_Monte_Carlo(object):
    """The Hybrid (Hamiltonian) Monte Carlo method
    
    Optional Inputs
        store_acceptance :: bool :: optionally store the acceptance rates
    
    Required Inputs
        x0         :: tuple :: initial starting position vector
        potential  :: class :: class from potentials.py
        dynamics   :: class :: integrator class for dynamics from h_dynamics.py
        rng        :: np.random.RandomState :: random number state
    Expectations
    """
    def __init__(self, x0, dynamics, potential, rng, store_acceptance=False):
        
        self.x0 = x0
        self.dynamics = dynamics
        self.potential = potential
        self.rng = rng
        self.store_acceptance = store_acceptance
        
        self.momentum = Momentum(self.rng)
        self.accept = Accept_Reject(self.rng, store_acceptance=store_acceptance)
        
        self.x = self.x0
        
        # Take the position in just for the shape
        # as note this is a fullRefresh so the return is
        # entirely gaussian noise. It is independent of X
        # so no need to preflip X before using as an input
        self.p = self.momentum.fullRefresh(self.x0) # intial mom. sample
        
        shapes = (self.x.shape, self.p.shape)
        checks.tryAssertEqual(*shapes,
             error_msg=' x.shape != p.shape' \
             +'\n x: {}, p: {}'.format(*shapes)
             )
        
        pass
    
    def sample(self, n_samples, n_burn_in = 20, mixing_angle=.5*np.pi, verbose = False):
        """runs the sampler for HMC
        
        Required Inputs
            n_samples   :: integer :: Number of samples (# steps after burn in)
        
        Optional Inputs
            n_burn_in   :: integer :: Number of steps to discard at start
            store_path  :: bool    :: Store path for plotting
            mixing_angle :: float    :: 0 is no mixing, pi/2 is total mix
            verbose :: bool :: a progress bar if True
        
        Returns
            (p_data, x_data) where *_data = (burn in samples, sample)
        """
        self.burn_in_p = [self.p.copy()]
        self.burn_in = [self.x.copy()]
        
        iterator = range(n_burn_in)
        if verbose: 
            print '\nBurning in ...'
            iterator = tqdm(iterator)        
        
        for step in iterator: # burn in
            self.p, self.x = self.move(mixing_angle=mixing_angle)
            self.burn_in_p.append(self.p.copy())
            self.burn_in.append(self.x.copy())
        
        self.samples_p = [self.p.copy()]
        self.samples = [self.x.copy()]
        
        iterator = range(n_samples)
        if verbose: 
            print 'Sampling ...'
            iterator = tqdm(iterator)
        
        for step in iterator:
            p, x = self.move(mixing_angle=mixing_angle)
            self.samples_p.append(self.p.copy())
            self.samples.append(self.x.copy())
        
        return (self.burn_in_p, self.samples_p), (self.burn_in, self.samples)
    
    def move(self, step_size = None, n_steps = None, mixing_angle=.5*np.pi):
        """A generalised Hybrid Monte Carlo move:
        Combines Hamiltonian Dynamics and Momentum Refreshment
        to generate a the next position for the MCMC chain
        
        As HMC but includes option for partial momentum refreshment
        through the mixing angle
        
        Optional Inputs
            step_size    :: float    :: step_size for integrator
            n_steps      :: integer  :: number of integrator steps
            mixing_angle :: float    :: 0 is no mixing, pi/2 is total mix
        Returns
            (p,x) :: (float, float) :: new momentum and position
        """
        
        # Determine current energy state
        # p,x = self.p.copy(), self.x.copy()
        h_old = self.potential.hamiltonian(self.p, self.x)    # get old hamiltonian
        
        # The use of indices emphasises that the
        # mixing happens point-wise
        p = np.zeros(self.p.shape)
        for idx in np.ndindex(self.p.shape):
            # although a flip is added when theta=pi/2
            # it doesn't matter as noise is symmetric
            p[idx] = self.momentum.generalisedRefresh(p[idx], mixing_angle=mixing_angle)
        
        # Molecular Dynamics Monte Carlo
        if (step_size is not None): self.dynamics.step_size = step_size
        if (n_steps is not None): self.dynamics.n_steps = step_size
        p, x = self.dynamics.integrate(self.p, self.x)
        
        # # GHMC flip if partial refresh - else don't bother.
        # if (mixing_angle != .5*np.pi):
        p = self.momentum.flip(p)
        
        # Metropolis-Hastings accept / reject condition
        h_new = self.potential.hamiltonian(p, x) # get new hamiltonian
        accept = self.accept.metropolisHastings(h_old=h_old, h_new=h_new)
        
        # If the proposed state is not accepted (i.e. it is rejected), 
        # the next state is the same as the current state 
        # (and is counted again when estimating the expectation of 
        # some function of state by its average over states of the Markov chain)
        # - Neal, "MCMC using Hamiltnian Dynamics"
        # checks.tryAssertNotEqual(h_old, h_new,
        #      error_msg='H has not changed!' \
        #      +'\n h_old: {}, h_new: {}'.format(h_old, h_new)
        #      )
        # checks.tryAssertNotEqual(x,self.x,
        #      error_msg='x has not changed!' \
        #      +'\n x: {}, self.x: {}'.format(x, self.x)
        #      )
        # checks.tryAssertNotEqual(p,self.p,
        #      error_msg='p has not changed!' \
        #      +'\n p: {}, self.p: {}'.format(p, self.p)
        #      )
        
        if not accept: p, x = self.p, self.x # return old p,x
        return p,x
    
#
class Momentum(object):
    """Momentum Routines
    
    Required Inputs
        rng :: np.random.RandomState :: random number generator
    """
    def __init__(self, rng):
        self.rng = rng
        pass
    
    def fullRefresh(self, p):
        """Performs full refresh with implicit mixing angle
        
        Required Inputs
            p :: np.array :: momentum to refresh
        """
        p = self.generalisedRefresh(p)
        return p
    
    def generalisedRefresh(self, p, mixing_angle=.5*np.pi):
        """Performs partial refresh through mixing angle
        to mix momentum with gaussian noise
        
        Optional Inputs
            mixing_angle :: float   :: 0 is no mixing, pi/2 is total mix
        
        Required Inputs
            p :: np.array :: momentum to refresh
        """
        
        # Random Gaussian noise with: sdev=scale & mean=loc
        self.noise = self.rng.normal(size=p.shape, scale=1., loc=0.)
        self.mixed = self._refresh(p, self.noise, theta=mixing_angle)
        
        ret_val = np.asarray(self.mixed[:p.size, :1]).reshape(p.shape)
        return ret_val
    
    def _refresh(self, p, noise, theta):
        """Mixes noise with momentum
        
        Required Inputs
            p       :: np.array :: momentum to refresh
            noise   :: np.array :: noise to mix with momentum
            theta   :: float    :: mixing angle
        """
        
        self.rot = self._rotationMatrix(n_dim=p.size, theta=theta)
        
        p = np.matrix(p.ravel()).T          # column A
        noise = np.matrix(noise.ravel()).T  # column B
        
        # bmat is block matrix format
        unmixed = np.bmat([[p],[noise]])    # column [[A],[B]]
        flipped = self.flip(unmixed)
        
        # matix multiplication
        mixed = self.rot*flipped
        
        return mixed
    
    def _rotationMatrix(self, n_dim, theta):
        """A rotation matrix
        
        Required Input
            theta :: float :: angle in radians for mixing
            n_dim :: tuple :: creates n_dim^2 blocks (total: 4*n_dim^2)
        
        Expectations
            This heavily relies on the np.matrix formatting
            as defined in _refresh(). It will NOT work with
            the standard np.ndarray class
        """
        i = np.identity(n_dim)
        c, s = np.cos(theta)*i, np.sin(theta)*i
        rotation = np.bmat([[c, s], [-s, c]])
        return rotation
    
    def flip(self, p):
        """Reverses the momentum
        
        Required Inputs
            p       :: np.array :: momentum to refresh
        """
        return -p
    
    
#
if __name__ == '__main__':
    pass