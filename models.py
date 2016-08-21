import numpy as np
from hmc.lattice import Periodic_Lattice
from hmc.hmc import *
from hmc.common import Init

class Base(object):
    def __init__(self):
        pass
    
    def run(self, n_samples, n_burn_in, **kwargs):
        """Runs the HMC algorithm to sample the potential
        
        Required Inputs
            n_samples   :: int  :: number of samples
            n_burn_in   :: int  :: number of burnin steps
        
        Optional Inputs
            mixing_angle :: float :: mixing angle for sampler
            verbose :: bool :: a progress bar if True
            any parameter that can be passed to self.sampler.sample()
        """
        p_samples, samples = self.sampler.sample(
            n_samples = n_samples, n_burn_in = n_burn_in, **kwargs)
        burn_in, samples = samples # return the shape: (n, dim, 1)
        traj = self.sampler.samples_traj
        
        # flatten last dimension to a shape of (n, dim)
        self.burn_in = np.asarray(burn_in).reshape(n_burn_in+1, -1)
        self.samples = np.asarray(samples).reshape(n_samples+1, -1)
        self.traj  = np.asarray(traj, dtype='float64').reshape(n_samples+1)
        self.traj *= self.step_size
        self.p_acc = np.asarray(self.sampler.accept.accept_rates).ravel().mean()
        self.p_acc = np.asscalar(self.p_acc)
        pass
    
    def _getInstances(self):
        """gets the relevant instances for the model"""
        
        self.x0 = Periodic_Lattice(self.x0, lattice_spacing=self.spacing)
        if not hasattr(self, 'save_path'): self.save_path=False
        if not hasattr(self, 'save_path'): self.save_path=False
        dynamics = Leap_Frog(
            duE = self.pot.duE,
            step_size = self.step_size,
            n_steps = self.n_steps,
            rand_steps = self.rand_steps,
            save_path = self.save_path)
        
        if hasattr(self, 'accept_kwargs'):
            if 'get_accept_rates' not in self.accept_kwargs:
                self.accept_kwargs['get_accept_rates'] = True
        else:
            self.accept_kwargs = {'get_accept_rates':True}
        
        self.sampler = Hybrid_Monte_Carlo(self.x0, dynamics, self.pot, self.rng,
            accept_kwargs = self.accept_kwargs)
        pass
#
class Basic_HMC(Init, Base):
    """A HMC model to sample from the potentials with LeapFrog
    
    Required Inputs
        x0          :: position (lattice)
        pot         :: potential class - see hmc.potentials
    
    Optional Inputs
        n_steps     :: int  :: default number of steps for dynamics
        step_size   :: int  :: default step size for dynamics
        spacing     :: float :: lattice spacing
        rng :: np.random.RandomState :: must be able to call rng.uniform
    """
    def __init__(self, x0, pot, **kwargs):
        super(Basic_HMC, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'spacing':1.,
            'rng':np.random.RandomState(111),
            'step_size': .1,
            'n_steps': 20,
            'rand_steps':False
        }
        
        self.initDefaults(kwargs)
        self._getInstances()
        pass
    
#
class Basic_KHMC(Init, Base):
    """A KHMC model to sample from the potentials with LeapFrog
    
    Required Inputs
        x0          :: position (lattice)
        pot         :: potential class - see hmc.potentials
    
    Optional Inputs
        step_size   :: int  :: default step size for dynamics
        spacing     :: float :: lattice spacing
        rng :: np.random.RandomState :: must be able to call rng.uniform
    """
    def __init__(self, x0, pot, **kwargs):
        super(Basic_KHMC, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'spacing':1.,
            'rng':np.random.RandomState(111),
            'step_size': .1,
            'rand_steps':False
        }
        self.initDefaults(kwargs)
        self.n_steps = 1 # this is a key paramter of KHMC
        self._getInstances()
        pass
    
#
class Basic_GHMC(Init, Base):
    """A GHMC model to sample from the potentials with LeapFrog
    
    Required Inputs
        x0          :: position (lattice)
        pot         :: potential class - see hmc.potentials
    
    Optional Inputs
        n_steps     :: int  :: default number of steps for dynamics
        step_size   :: int  :: default step size for dynamics
        spacing     :: float :: lattice spacing
        rng :: np.random.RandomState :: must be able to call rng.uniform
    """
    def __init__(self, x0, pot, **kwargs):
        super(Basic_GHMC, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'spacing':1.,
            'rng':np.random.RandomState(111),
            'step_size': .1,
            'n_steps': 20,
            'rand_steps':False
        }
        self.initDefaults(kwargs)
        self._getInstances()
        pass
    