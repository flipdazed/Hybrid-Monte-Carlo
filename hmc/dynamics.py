import numpy as np
from tqdm import tqdm

import checks

class Leap_Frog(object):
    """Leap Frog Integrator
    
    Required Inputs
        duE  :: func :: Gradient of Potential Energy
    
    Optional Inputs
        step_size   :: integration step size
        n_steps     :: leap frog integration steps (trajectory length)
        save_path   :: saves the integration path - see _stepSteps() for locations
    
    Note: Do not confuse x0,p0 with initial x0,p0 for HD
    """
    def __init__(self, duE, step_size = 0.1, n_steps = 250, save_path = False):
        self.step_size = step_size
        self.n_steps = n_steps
        self.duE = duE
        
        self.save_path = save_path
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
        p, x = fn(p0, x0, verbose = False)
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
        p, x = p0, x0
        self._storeSteps(p, x) # store zeroth step
        
        iterator = range(0, self.n_steps)
        if verbose: iterator = tqdm(iterator)
        for step in iterator:
            self._moveP(p, x, frac_step=0.5)
            self._moveX(p, x)
            self._moveP(p, x, frac_step=0.5)
            self._storeSteps(p, x) # store moves
        
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
        
        p, x = p0,x0
        self._moveP(p, x, frac_step=0.5)
        self._moveX(p, x)
        
        iterator = range(0, self.n_steps)
        if verbose: iterator = tqdm(iterator)
        for step in iterator:
            self._moveP(p, x)
            self._moveX(p, x)
        
        self._moveP(p, x, frac_step=0.5)
        
        return p, x
    
    def _moveX(self, p, x, frac_step = 1.):
        """Calculates a POSITION move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        
        x += frac_step*self.step_size*p
        pass
    
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
        for index in np.ndindex(p.shape):
            try:
                p[index] -= frac_step*self.step_size*self.duE(x, index)
            except:
                checks.fullTrace(msg='idx: {}, deriv {}'.format(index, self.duE(x, index)))
        pass
    def _storeSteps(self, p, x, new=False):
        """Stores current momentum and position in lists
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        
        Optional Input
            new :: bool :: reverts storage arrays to empty lists
        Expectations
            self.x_step :: float
            self.p_step :: float
        """
        self.p_ar.append(p.copy())
        self.x_ar.append(x.copy())
        pass
    
    def newPaths(self):
        """Initialises new path lists"""
        self.p_ar = [] # data for plots
        self.x_ar = [] # data for plots
        pass
#
if __name__ == '__main__':
    pass