import numpy as np
from copy import copy

class Leap_Frog(object):
    """Leap Frog Integrator
    
    Required Inputs
        duE  :: func :: Gradient of Potential Energy
    
    Optional Inputs
        d   :: integration step size
        l  :: leap frog integration steps (trajectory length)
    
    Note: Do not confuse x0,p0 with initial x0,p0 for HD
    """
    def __init__(self, duE, step_size = 0.1, n_steps = 250, lattice=False, save_path = False):
        self.step_size = step_size
        self.n_steps = n_steps
        self.duE = duE
        self.lattice = lattice
        
        self.save_path = save_path
        self.newPaths() # create blank lists
        pass
    
    def integrate(self, p0, x0):
        """The Leap Frog Integration
        
        Required Input
            p0  :: float :: initial momentum to start integration
            x0  :: float :: initial position to start integration
        
        Expectations
            save_path :: Bool :: save (p,x). IN PHASE: Start at (1,1)
            self.x_step = x0 when class is instantiated
            self.p_step = p0 when class is instantiated
        
        Returns
            (x,p) :: tuple :: momentum, position
        """
        self.p, self.x = p0, x0
        if self.save_path:
            self._storeSteps() # store zeroth step
        
        # print "     ... begin integration"
        for step in xrange(0, self.n_steps):
            # print "         > step: {}".format(step)
            # b = copy(self.x.get)
            self._moveP(frac_step=0.5)
            self._moveX()
            self._moveP(frac_step=0.5)
            # print self.p - b
            # print self.p
            # print self.x.get
            if self.save_path: self._storeSteps() # store moves
        # print "     ... end integration"
        
        # remember that any usage of self.p, self.x will be stored as a pointer
        # must slice or use a copy(self.p) to "freeze" the current value in mem
        return self.p, self.x
    
    def integrateAlt(self, p0, x0):
        """The Leap Frog Integration
        
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
        
        self.p, self.x = p0,x0
        self._moveP(frac_step=0.5)
        self._moveX()
        if self.save_path: self._storeSteps() # store moves
        
        for step in xrange(1, self.n_steps):
            self._moveP()
            self._moveX()
            if self.save_path: self._storeSteps() # store moves
        
        self._moveP(frac_step=0.5)
        
        return self.p, self.x
    
    def _moveX(self, frac_step = 1.):
        """Calculates a POSITION move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        # print "         > X Move"
        
        if not self.lattice:
            # all addition can be done in place as it is a one-to-one operation
            self.x += frac_step*self.step_size*self.p
            
        else: # lattice is present
            # all addition can be done in place as it is a one-to-one operation
            self.x.get += frac_step*self.step_size*self.p
        pass
    
    def _moveP(self, frac_step = 1.):
        """Calculates a MOMENTUM move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        
        Expectations
            self.p :: field in the first dimension, lattice in `self.p.shape[1:]`
        
        The momenta moves sweep through all the momentum field positions and
        update them in turn using a numpy array iteration
        
        The separation between lattice and non-lattice theory is due to the
        potentials used. The SHO and Multivariate Gaussian derivatives do not accept
        position indices for gradient terms and have an analytical derivative term.
        
        This needs to be corrected.
        """
        # print "         > P Move"
        
        if not self.lattice:
            self.p -= frac_step*self.step_size*self.duE(self.x)
            
        else: # lattice is present
            for index in np.ndindex(self.p.shape):
                self.p[index] -= frac_step*self.step_size*self.duE(self.x, index)
        pass
    def _storeSteps(self, new=False):
        """Stores current momentum and position in lists
        
        Optional Input
            new :: bool :: reverts storage arrays to empty lists
        Expectations
            self.x_step :: float
            self.p_step :: float
        """
        if not self.lattice:
            self.p_ar.append(copy(self.p))
            self.x_ar.append(copy(self.x))
        else: # lattice is present
            self.p_ar.append(copy(self.p))
            self.x_ar.append(copy(self.x.get))
        pass
    
    def newPaths(self):
        """Initialises new path lists"""
        self.p_ar = [] # data for plots
        self.x_ar = [] # data for plots
        pass
    def pathToNumpy(self):
        """converts the path to a numpy array
        
        make into a 3-tensor of (path, column x/p, 1)
        last shape required to preserve as column vector for np. matrix mul
        """
        self.p_ar = np.asarray(self.p_ar).reshape(
            (len(self.p_ar), self.p.shape[0], self.p.shape[1]))
        self.x_ar = np.asarray(self.x_ar).reshape(
            (len(self.x_ar), self.x.shape[0], self.x.shape[1]))
        pass
#
if __name__ == '__main__':
    pass