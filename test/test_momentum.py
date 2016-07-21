import numpy as np

import utils

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.hmc import Momentum

class Test(object):
    """Tests for the HMC class
    
    Required Inputs
        rng :: np.random.RandomState :: random number generator
    """
    def __init__(self, rng):
        self.rng = rng
        self.m = Momentum(rng)
        pass
    
    def mixing(self, print_out):
        """Tests the mixing matrix
        
        Required Input
            print_out :: bool :: print results to screen
        """
        passed = True
        def create_display(self, test_name, p, expected, passed):
            utils.display(test_name = test_name, 
            outcome = passed,
            details = {
                'original array: {}'.format(np.bmat([[p],[self.noise]])):[],
                #'rotation matrix: {}'.format(self.m.rot):[],
                'mix + flip: {}'.format(self.mixed):[],
                'expected: {}'.format(expected):[]
                })
            pass
        
        p14 = np.asarray([[1.,7.,-1., 400.]])
        self.noise = np.asarray([[0.1, 3., 1., -2]])
        
        expected_0 = np.asarray([[-1., -7., 1., -400.,-0.1, -3., -1., 2.]])
        expected_halfpi = np.asarray([[-0.1, -3., -1., 2., 1., 7., -1., 400.]])
        expected_pi = -expected_0
        expected_quartpi = np.asarray( [[-0.778, -7.071,  0.,   -281.428,
                                      0.636,  2.828, -1.414, 284.257]])
        
        expected_0 = expected_0.flatten()[:4]
        expected_halfpi = expected_halfpi.flatten()[:4]
        expected_pi = expected_pi.flatten()[:4]
        expected_quartpi = expected_quartpi.flatten()[:4]
        
        self.mixed = self.m._refresh(p14, self.noise, theta = 0.)
        passed = (np.around(self.mixed, 3) == expected_0).all()
        if print_out: 
            create_display(self,
                "Theta = 0. (No Mix)", p14, expected_0, passed)
        
        self.mixed = self.m._refresh(p14, self.noise, theta = np.pi)
        passed = (np.around(self.mixed, 3) == expected_pi).all()
        if print_out:
            create_display(self,
                "Theta = pi (- No Mix)", p14, expected_pi, passed)
        
        self.mixed = self.m._refresh(p14, self.noise, theta = np.pi/2.)
        passed = (np.around(self.mixed, 3) == expected_halfpi).all()
        if print_out:
            create_display(self, "Theta = pi/2. (Total Mix)",
                p14, expected_halfpi, passed)
        
        self.mixed = self.m._refresh(p14, self.noise, theta = np.pi/4.)
        passed = (np.around(self.mixed, 3) == expected_quartpi).all()
        if print_out:
            create_display(self,
                "Theta = pi/4. (Total Mix)", p14, expected_quartpi, passed)
            
        return passed
    
    def vectors(self, p, print_out):
        """Tests different momentum vector shape for correct mixing
        
        Required Inputs
            p           :: np.array :: momentum to refresh
            test_name   :: string   :: name of test
            print_out   :: bool     :: print results to screen
        """
        
        p_mixed = self.m.fullRefresh(p=p)
        passed = (np.around(self.m.flip(p_mixed), 6) == np.around(self.m.noise, 6)).all()
        
        if print_out:
            utils.display(test_name='full refresh: {} dimensions'.format(p.shape), 
            outcome = passed,
            details = {
                'original array: {}'.format(np.bmat([[p],[self.m.noise]])):[],
                # 'rotation matrix: {}'.format(self.m.rot):[],
                'mix + flip: {}'.format(p_mixed):[],
                })
        
        return passed
    
if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    test = Test(rng=rng)

    rand4 = np.random.random(4)
    p41 = np.mat(rand4.reshape(4,1))
    p22 = np.mat(rand4.reshape(2,2))
    p14 = np.mat(rand4.reshape(1,4))
    
    assert test.vectors(p41, print_out = True)
    assert test.vectors(p22, print_out = True)
    assert test.vectors(p14, print_out = True)
    
    assert test.mixing(print_out = True)