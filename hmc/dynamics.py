import numpy as np
from tqdm import tqdm

from common import Init
import checks

class Leap_Frog(Init):
    """Leap Frog Integrator
    
    Required Inputs
        duE  :: func :: Gradient of Potential Energy
    
    Optional Inputs
        step_size   :: integration step size
        n_steps     :: leap frog integration steps (trajectory length)
        save_path   :: saves the integration path - see _stepSteps() for locations
    
    Note: Do not confuse x0,p0 with initial x0,p0 for HD
    """
    def __init__(self, duE, **kwargs):
        super(Leap_Frog, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'step_size':0.1,
            'n_steps':250,
            'rand_steps':False,
            'save_path':False
            }
        self.initDefaults(kwargs)
        self.lengths = []
        self.newPaths() # create blank lists
        pass
    
    def integrate(self, p0, x0, verbose = False):
        """The Leap Frog Integration: optimises method
        
        Required Input
            p0  :: float :: initial momentum to start integration
            x0  :: float :: initial position to start integration
        
        Optional Input
            verbose :: bool :: prints out progress bar if True (ONLY use for LARGE path lengths)
        
        Expectations
            save_path :: Bool :: save (p,x). IN PHASE: Start at (1,1)
            self.x_step = x0 when class is instantiated
            self.p_step = p0 when class is instantiated
        
        Returns
            (x,p) :: tuple :: momentum, position
        """
        if self.save_path:
            fn = getattr(self, '_integrateSave')
        else:
            fn = getattr(self, '_integrateFast')
        
        p, x = fn(p0, x0, verbose = verbose)
        return p, x
    
    def _integrateSave(self, p0, x0, verbose = False):
        """The Leap Frog Integration - optimised for saving data
        
        Required Input
            p0  :: float :: initial momentum to start integration
            x0  :: float :: initial position to start integration
        
        Optional Input
            verbose :: bool :: prints out progress bar if True (ONLY use for LARGE path lengths)
        
        Expectations
            save_path :: Bool :: save (p,x). IN PHASE: Start at (1,1)
            self.x_step = x0 when class is instantiated
            self.p_step = p0 when class is instantiated
        
        Returns
            (x,p) :: tuple :: momentum, position
        """
        self.n = self._getStepLen()
        
        p, x = p0, x0
        self._storeSteps(p, x, self.n) # store zeroth step
        
        iterator = range(0, self.n)
        if verbose: iterator = tqdm(iterator)
        for step in iterator:
            p = self._moveP(p, x, frac_step=0.5)
            x = self._moveX(p, x)
            p = self._moveP(p, x, frac_step=0.5)
            self._storeSteps(p, x, self.n) # store moves
        
        # remember that any usage of self.p, self.x will be stored as a pointer
        # must slice or use a self.p.copy() to "freeze" the current value in mem
        return p, x
    
    def _integrateFast(self, p0, x0, verbose = False):
        """The Leap Frog Integration - a faster implementation
        
        Required Input
            p0  :: float :: initial momentum to start integration
            x0  :: float :: initial position to start integration
        
        Expectations
            save_path :: Bool :: save (p,x). OUT OF PHASE: Start at (.5,1)
            self.x_step = x0 when class is instantiated
            self.p_step = p0 when class is instantiated
        
        Returns
            (x,p) :: tuple :: momentum, position
        """
        self.n = self._getStepLen()
        
        # first step and half momentum step
        p = self._moveP(p0, x0, frac_step=0.5)
        x = self._moveX(p, x0)
        
        iterator = range(1, self.n) # one step done
        if verbose: iterator = tqdm(iterator)
        for step in iterator:
            p = self._moveP(p, x)
            x = self._moveX(p, x)
        
        # last half momentum step
        p = self._moveP(p, x, frac_step=0.5)
        
        return p, x
    
    def _getStepLen(self):
        """Determines if steps are constant or binomially distributed
        
        Obtains the number of steps from binomial trials using geometric dist.
        if binomially distributed (negative binomial dist.)
        """
        if not self.rand_steps:
            return self.n_steps
        else:
            return np.random.geometric(1./float(self.n_steps))
    
    def _moveX(self, p, x, frac_step = 1.):
        """Calculates a POSITION move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        
        x += frac_step*self.step_size*p
        return x
    
    def _moveP(self, p, x, frac_step = 1.):
        """Calculates a MOMENTUM move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        
        Expectations
            p :: field in the first dimension, lattice in `p.shape[1:]`
        
        The momenta moves sweep through all the momentum field positions and
        update them in turn using a numpy array iteration
        """
        # the extra value in the case of a non lattice potential is
        # garbaged by *args
        # for index in np.ndindex(p.shape):
        try:
            p -= frac_step*self.step_size*self.duE(x)
            # p[index] -= frac_step*self.step_size*self.duE(x, index)
        except:
            checks.fullTrace(msg='idx: {}, deriv {}'.format(index, self.duE(x, index)))
        return p
    
    def _storeSteps(self, p, x, l):
        """Stores current momentum and position in lists
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
            l :: int   :: number of steps in this trajectory
        
        Expectations
            self.x_step :: float
            self.p_step :: float
        """
        self.p_ar.append(p.copy())
        self.x_ar.append(x.copy())
        self.lengths.append(l)
        pass
    
    def newPaths(self):
        """Initialises new path lists"""
        self.p_ar = [] # data for plots
        self.x_ar = [] # data for plots
        self.n    = []
        pass
#
if __name__ == '__main__':
    pass