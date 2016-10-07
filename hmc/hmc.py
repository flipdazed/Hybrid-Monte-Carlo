# default pip imports
import numpy as np
from tqdm import tqdm

# local imports
import checks
from common import Init
from dynamics import Leap_Frog
from metropolis import Accept_Reject

__docformat__ = "restructuredtext en"

class Hybrid_Monte_Carlo(Init):
    """The Hybrid (Hamiltonian) Monte Carlo method
    
    This class contains the framework to extract various observables from the
    Generalised Hybrid Monte Carlo model
    
    Parameters
    ----------
    x0         : array_like
        Initial starting position vector
    potential  : class
        A potential class following the structure
        in :mod:`potentials`
    dynamics   : class
        Integrator class for Hamiltonian Dynamics following the structure
        in :mod:`dynamics`
    rng        : `np.random.RandomState`
        random number state
    store_acceptance : bool, optional
        optionally store the acceptance rates
    accept_kwargs    : dict, optional
        A dictionary of keyworkd arguments (kwargs) to pass to
        :class:`metropolis.Accept_Reject` as `self.accept(**accept_kwargs)`
    
    Methods
    ----------
    sample
        Runs the MCMC sampler. Returns both momentum and position lattices, 
        `(burn_in_p, samples_p), (burn_in, samples)`
    move
        Makes a GHMC move. Returns `p,x` if accepted and `p0,x0` if not.
    
    Attributes
    ----------
    dynamics
        An instance of the chosen integrator (e.g. :class:`dynamics.Leap_Frog`)
    momentum
        An instance of `Momentum` containing momentum refreshment maps
    accept
        An instance of :class:`metropolis.Accept_Reject` instanciated with `accept_kwargs`
    p0
        Initial momentum
    x0 
        Initial position
    
    """
    def __init__(self, x0, dynamics, potential, rng, **kwargs):
        super(Hybrid_Monte_Carlo, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'accept_kwargs':{ # kwargs to pass to accept
                'store_acceptance':False
                }
            }
        self.initDefaults(kwargs)
        
        # legacy support - need to update
        a = 'store_acceptance'
        if a in kwargs: self.accept_kwargs[a] = kwargs[a]
        
        self.momentum = Momentum(self.rng)
        self.accept = Accept_Reject(self.rng, **self.accept_kwargs)
        
        # Take the position in just for the shape
        # as note this is a fullRefresh so the return is
        # entirely gaussian noise. It is independent of X
        # so no need to preflip X before using as an input
        self.p0 = self.momentum.fullRefresh(self.x0) # intial mom. sample
        shapes = (self.x0.shape, self.p0.shape)
        
        checks.tryAssertEqual(*shapes,
             error_msg=' x0.shape != p0.shape' \
             +'\n x0: {}, p0: {}'.format(*shapes))
        self.h_old = None
        pass
    
    def sample(self, n_samples, n_burn_in = 20, mixing_angle=.5*np.pi, verbose = False, verb_pos = 0):
        """Runs the sampler for GHMC
        
        Parameters
        ----------
        n_samples       : integer
            Number of samples (# steps after burn in)
        n_burn_in       : int,  optional 
            Number of steps to discard at start
        store_path      : bool, optional 
            Store path for plotting
        mixing_angle    : float,optional 
            `0` is no mixing, pi/2 is total mix. The mixing angle is implemented so that 
            it could be converted to a function that varies the angle with respect to 
            some input
        verbose         : bool, optional 
            A progress bar if True
        verb_pos        : int,  optional 
            Offset for status bar
        
        Notes
        ----------
        Returns  `(p_data, x_data) where *_data = (burn in samples, sample)`
        although it is expected that these values will be extracted as class 
        attributes
        """
        p, x = self.p0.copy(), self.x0.copy()
        self.h_old = None
        
        # Burn in section
        self.burn_in_p = [p.copy()]
        self.burn_in = [x.copy()]
        self.burn_in_traj = [0]
        
        iterator = xrange(n_burn_in)
        for step in iterator: # burn in
            p, x = self.move(p, x, mixing_angle=mixing_angle)
            self.burn_in_p.append(p.copy())
            self.burn_in.append(x.copy())
            self.burn_in_traj.append(self.dynamics.n)
        
        # sampling section
        self.samples_p = [p.copy()]
        self.samples = [x.copy()]
        self.samples_traj = [0]
        
        iterator = xrange(n_samples)
        if verbose:
            iterator = tqdm(iterator, position=verb_pos, 
                desc='Sampling: {}'.format(verb_pos))
            # tqdm.write('Sampling ...')
        for step in iterator:
            p, x = self.move(p, x, mixing_angle=mixing_angle)
            self.samples_p.append(p.copy())
            self.samples.append(x.copy())
            self.samples_traj.append(self.dynamics.n)
        
        return (self.burn_in_p, self.samples_p), (self.burn_in, self.samples)
    
    def move(self, p, x, step_size = None, n_steps = None, mixing_angle=.5*np.pi):
        """A generalised Hybrid Monte Carlo move:
        Combines Hamiltonian Dynamics and Momentum Refreshment
        to generate a the next position for the MCMC chain
        
        As HMC but includes option for partial momentum refreshment
        through the mixing angle
        
        Parameters
        ----------
        step_size    : float,   optional
            Step_size for integrator
        n_steps      : integer, optional
            Number of integrator steps
        mixing_angle : float,   optional
            `0` is no mixing, `np.pi/2.` is a total refreshment
        
        Notes:
        If the proposed state is not accepted (i.e. it is rejected),
        the next state is the same as the current state
        (and is counted again when estimating the expectation of
        some function of state by its average over states of the Markov 
        chain) :cite:`Neal2011a`
        
        .. bibliography:: references.bib
        """
        if (step_size is not None): self.dynamics.step_size = step_size
        if (n_steps is not None): self.dynamics.n_steps = step_size
        
        # although a flip is added when theta=pi/2 it doesn't matter as noise is symmetric
        p = self.momentum.generalisedRefresh(p, mixing_angle=mixing_angle)
        
        # Determine current energy state
        p0, x0 = p.copy(), x.copy()
        
        # Molecular Dynamics Monte Carlo
        p, x = self.dynamics.integrate(p, x)
        
        # # GHMC flip if partial refresh - else don't bother.
        # if (mixing_angle != .5*np.pi):
        p = self.momentum.flip(p)
        
        # Metropolis-Hastings accept / reject condition
        self.h_old = self.potential.hamiltonian(p0, x0)     # get old hamiltonian (after mom refresh)
        self.h_new = self.potential.hamiltonian(p, x)       # get new hamiltonian
        accept = self.accept.metropolisHastings(h_old=self.h_old, h_new=self.h_new)
        
        if accept: return p,x
        else: return p0, x0 # return old p,x
    
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
        
        return self.mixed
    
    def _refresh(self, p, noise, theta):
        """Mixes noise with momentum
        
        Required Inputs
            p       :: np.array :: momentum to refresh
            noise   :: np.array :: noise to mix with momentum
            theta   :: float    :: mixing angle
        """
        if theta == .5*np.pi:
            return self.flip(noise)
        elif theta == 0:
            return self.flip(p)
        else:
            return np.cos(theta)*self.flip(p) + np.sin(theta)*self.flip(noise)
    
    def _refreshOld(self, p, noise, theta):
        """Mixes noise with momentum
        
        ### This is now retired from use as a faster implementation is
        in _refresh() - we never need the gaussian noise back
        
        Required Inputs
            p       :: np.array :: momentum to refresh
            noise   :: np.array :: noise to mix with momentum
            theta   :: float    :: mixing angle
        """
        if theta == .5*np.pi:
            return self.flip(noise)
        elif theta == 0:
            return self.flip(p)
        else:
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
        
        ### Retired with _refreshOld()
        
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
