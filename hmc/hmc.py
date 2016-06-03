import numpy as np
from potentials import Simple_Harmonic_Oscillator
from h_dynamics import Hamiltonian_Dynamics, Leap_Frog

class Hybrid_Monte_Carlo(object):
    """The Hybrid (Hamiltonian) Monte Carlo method
    
    Optional Inputs
        
    
    Required Inputs
        
    
    Expectations
        
    """
    def __init__(self, samples, p0, x0):
        self.samples = samples
        self.p0, self.x0 = p0, x0
        
        self.potential = Simple_Harmonic_Oscillator()
        self.integrator = Leap_Frog(duE=self.potential.duE, d=0.3, l = 20)
        self.momentum = Momentum()
        pass
    
    def sample(self):
        
        pass
    
class Momentum(object):
    def __init__(self):
        pass
    def fullRefresh(self, p):
        """Performs full refresh"""
        p = self.generalisedRefresh()
        return p
    def generalisedRefresh(self, p, mixing_angle=None):
        """Performs partial refresh through mixing angle
        
        Optional Inputs
            mixing_angle :: float :: 0. is no mixing
        
        Required Inputs
            p :: np.array :: momentum to refresh
        """
        if mixing_angle is None: mixing_angle = 2*np.pi
        
        # Random Gaussian noise with: sdev=scale & mean=loc
        noise = np.random.normal(size=p.shape, scale=1., loc=0.)
        r = self._rotationMatrix(theta=mixing_angle)
        
        unmixed = np.matrix([[p],[noise]])
        flipped = self.flip(unmixed)
        
        # note that this is matix multiplication
        mixed = r*flipped
        return p
    
    def _rotationMatrix(self, theta):
        """A rotation matrix
        
        Required Input
            theta :: float :: angle in radians for mixing
        """
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.matrix([[c, s], [-s, c]])
        return rotation
    
    def flip(self):
        """Reverses the momentum"""
        self.p *= -1.
        pass
#
class Test(object):
    """Tests to verify the HMC algorithm"""
    def __init__(self):
        pass
    def BivariateNormal(self):
        """A test to sample the Bivariate Normal Distribution"""
        pass
    
#
if __name__ == '__main__':
    
    test = Test()
    test.BivariateNormal()