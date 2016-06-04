import numpy as np
from potentials import Simple_Harmonic_Oscillator
from h_dynamics import Hamiltonian_Dynamics, Leap_Frog

class Hybrid_Monte_Carlo(object):
    """The Hybrid (Hamiltonian) Monte Carlo method
    
    Optional Inputs
        
    
    Required Inputs
        x0  :: n-dim array :: positions are 
        rng :: random number generator
    Expectations
        
    """
    def __init__(self, x0):
        self.samples = samples
        self.p0, self.x0 = p0, x0
        
        # These can be made into inputs
        self.rng = np.random.RandomState(1234)
        
        self.potential = Simple_Harmonic_Oscillator()
        self.integrator = Leap_Frog(duE=self.potential.duE, d=0.3, l = 20)
        
        self.momentum = Momentum()
        # End
        
        pass
    
    def move(self):
        
        pass
    
    def test():
        """A test to sample the Bivariate Normal Distribution"""
        pass
    
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
        """Performs full refresh"""
        p = self.generalisedRefresh(p)
        return p
    
    def generalisedRefresh(self, p, mixing_angle=.5*np.pi):
        """Performs partial refresh through mixing angle
        to mix momentum with gaussian noise
        
        Optional Inputs
            mixing_angle :: float :: 0. is no mixing
        
        Required Inputs
            p :: np.array :: momentum to refresh
        """
        
        # Random Gaussian noise with: sdev=scale & mean=loc
        self.noise = self.rng.normal(size=p.shape, scale=1., loc=0.)
        self.mixed = self._refresh(p, self.noise, theta=mixing_angle)
        
        return self.mixed[:p.shape[0],:p.shape[1]]
    
    def _refresh(self, p, noise, theta):
        """Mixes noise with momentum
        
        Required Inputs
            p       :: np.array :: momentum to refresh
            noise   :: np.array :: noise to mix with momentum
            theta   :: float    :: mixing angle
        """
        
        self.rot = self._rotationMatrix(n_dim=p.shape[0], theta=theta)
        
        unmixed = np.bmat([[p],[noise]])
        flipped = self.flip(unmixed)
        
        # matix multiplication
        mixed = self.rot*flipped
        
        return mixed
    
    def _rotationMatrix(self, n_dim, theta):
        """A rotation matrix
        
        Required Input
            theta :: float :: angle in radians for mixing
            n_dim :: tuple :: creates n_dim^2 blocks (total: 4*n_dim^2)
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
    
    def test(self, print_out=False):
        """tests the momentum refreshment
        
        Required Input
            print_out :: bool :: print results to screen
        
        1. tests 3 input momentum shapes
        2. tests 4 mixing angles
        """
        
        if print_out: np.set_printoptions(precision=2, suppress=True)
        
        passed = True
        rng =  np.random.RandomState(1234)
        
        rand4 = np.random.random(4)
        p41 = np.mat(rand4.reshape(4,1))
        p22 = np.mat(rand4.reshape(2,2))
        p14 = np.mat(rand4.reshape(1,4))
        
        passed *= self._testVectors(p41, "4x1 vector + fullRefresh()", print_out)
        passed *= self._testVectors(p22, "2x2 vector + fullRefresh()", print_out)
        passed *= self._testVectors(p14, "1x4 vector + fullRefresh()", print_out)
        
        passed *= self._testMixing(print_out)
        return passed
    
    def _testMixing(self, print_out):
        """Tests the mixing matrix
        
        Required Input
            print_out :: bool :: print results to screen
        """
        
        def display(test_name, p, expected, result): # print to terminal
            print '\n\n TEST: {}'.format(test_name)
            print '\n original:\n',np.bmat([[p],[self.noise]])
            print '\n rotation matrix:\n',self.rot
            print '\n mix + flip:\n',self.mixed
            print '\n expected:\n',expected
            print '\n outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        p14 = np.mat([[1.,7.,-1., 400.]])
        noise = np.mat([[0.1, 3., 1., -2]])
        
        expected_0 = np.mat([[-1., -7., 1., -400.], [-0.1, -3., -1., 2.]])
        expected_halfpi = np.mat([[-0.1, -3., -1., 2.], [1., 7., -1., 400.]])
        expected_pi = -expected_0
        expected_quartpi = np.mat( [[-0.778, -7.071, 0., -281.428],
                                    [0.636, 2.828, -1.414, 284.257]])
        
        self.mixed = self._refresh(p14, noise, theta = 0.)
        result = (np.around(self.mixed, 3) == expected_0).all()
        if print_out: display("Theta = 0. (Total Mix)", p14, expected_0, result)
        
        self.mixed = self._refresh(p14, noise, theta = np.pi)
        result = (np.around(self.mixed, 3) == expected_pi).all()
        if print_out: display("Theta = pi (- Total Mix)", p14, expected_pi, result)
        
        self.mixed = self._refresh(p14, noise, theta = np.pi/2.)
        result = (np.around(self.mixed, 3) == expected_halfpi).all()
        if print_out: display("Theta = pi/2. (No Mix)", p14, expected_halfpi, result)
        
        self.mixed = self._refresh(p14, noise, theta = np.pi/4.)
        result = (np.around(self.mixed, 3) == expected_quartpi).all()
        if print_out: display("Theta = pi/4. (Partial Mix)", p14, expected_quartpi, result)
        
        return result
    
    def _testVectors(self, p, test_name, print_out):
        """Tests different momentum vector shape for correct mixing
        
        Required Inputs
            p           :: np.array :: momentum to refresh
            test_name   :: string   :: name of test
            print_out   :: bool     :: print results to screen
        """
        
        def display(test_name, p, p_mixed, result): # print to terminal
            print '\n\n TEST: {}'.format(test_name)
            print '\n original vector:\n',np.bmat([[p],[self.noise]])
            print '\n rotation matrix:\n',self.rot
            print '\n mixed momentum + flip:\n',p_mixed
            print '\n outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        p_mixed = self.fullRefresh(p=p)
        result = (np.around(self.flip(p_mixed), 6) == np.around(self.noise, 6)).all()
        if print_out: display(test_name, p, p_mixed, result)
        
        return result
#
if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    m = Momentum(rng)
    print m.test(print_out=False)