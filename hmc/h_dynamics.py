class Leap_Frog(object):
    """Leap Frog Integrator
    
    Required Inputs
        duE  :: func :: Gradient of Potential Energy
    
    Optional Inputs
        d   :: integration step length
        lf  :: leap frog integration steps (trajectory length)
    
    Note: Do not confuse x0,p0 with initial x0,p0 for HD
    """
    def __init__(self, duE, d=0.1, l = 250, save_path=False):
        self.d = d
        self.l = l
        self.duE = duE
        
        self.save_path = save_path
        self.p_ar = [] # data for plots
        self.x_ar = [] # data for plots
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
        self.p, self.x = p0,x0
        for step in xrange(0, self.l):
            self._moveP(frac_step=0.5)
            self._moveX()
            self._moveP(frac_step=0.5)
            if self.save_path: self._storeSteps() # store moves
        
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
        
        for step in xrange(1, self.lf):
            self._moveP()
            self._moveX()
            if self.save_path: self._storeSteps() # store moves
        
        self._moveP(frac_step=0.5)
        
        return self.p, self.x
    
    def _moveX(self, frac_step=1.):
        """Calculates a POSITION move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        self.x += frac_step*self.d*self.p
        pass
    
    def _moveP(self, frac_step=1.):
        """Calculates a MOMENTUM move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        self.p -= frac_step*self.d*self.duE(self.x)
        pass
    def _storeSteps(self):
        """Stores current momentum and position in lists
        
        Expectations
            self.x_step :: float
            self.p_step :: float
        """
        self.p_ar.append(self.p)
        self.x_ar.append(self.x)
        pass


#
class Hamiltonian_Dynamics(object):
    """Simulates Hamiltonian Dynamics using arbitrary integrator
    
    Required Inputs
        p0  :: float :: initial momentum
        x0  :: float :: initial position
        integrator :: func :: integration function for HD
    
    Expectations
        Kinetic energy is a function of momentum
        Potential energy is a function of position
        Either may have additional parameters
    """
    def __init__(self, p0, x0, integrator):
        self.p0,self.x0 = p0, x0
        self.integrator = integrator
        
        self.p,self.x = p0,x0 # initial conditions
        pass
    
    def integrate(self):
        """Calculate a trajectory with Hamiltonian Dynamics"""
        self.p, self.x = self.integrator.integrate(self.p, self.x)
        pass