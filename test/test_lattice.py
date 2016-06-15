import numpy as np

import utils

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.lattice import Periodic_Lattice

TEST_ID = 'lattice'

class Test(object):
    def __init__(self):
        
        self.a1 = np.array([[ 11.,  12.,  13.,  14.],
                            [ 21.,  22.,  23.,  24.],
                            [ 31.,  32.,  33.,  34.],
                            [ 41.,  42.,  43.,  44.]])
        
        self.l = Periodic_Lattice(array=self.a1, spacing=1)
        
    def Wrap(self, print_out = True):
        """tests the wrapping function against expected values"""
        passed = True
        wi = self.l.wrapIdx
        a = self.a1 # shortcut
        passed *= (self.l.get == self.a1).all() # check both are equal
        test = [[(1,1), 22.], [(3,3), 44.], [(4,4), 11.], # [index, expected value]
            [(3,4), 41.], [(4,3), 14.], [(10,10), 33.]]
        
        for idx, act in test: # iterate test values
            passed *= (a[wi(index=idx)] == act)
        
        if print_out:
            minimal = (print_out == 'minimal')
            utils.display('Periodic Boundary', outcome=passed,
                details = {'array storage checked':[],
                    'period indexing vs. known values':[]},
                minimal = minimal)
        
        return passed
        
    def Laplacian(self, print_out = True):
        """tests the wrapping function against expected values"""
        passed = True
        a = self.a1 # shortcut
        passed *= (self.l.get == self.a1).all() # check both are equal
        test = [[(1,1), np.asarray([0., 0.])],
                [(3,3), np.asarray([-40., -4.])] ,
                [(4,4), np.asarray([40., 4.])], # [index, expected value]
                [(3,4), np.asarray([-40., 4.])],
                [(4,3), np.asarray([40., -4.])],
                [(2,3), np.asarray([0., -4.])]]
        
        for pos, act in test: # iterate test values
            res = self.l.laplacian(position=pos, a_power=0)
            passed *= (res == act).all()
        
        if print_out:
            minimal = (print_out == 'minimal')
            utils.display('Laplacian', outcome=passed,
                details = {'checked vs. known values (Mathematica)':['wrapped boundary values included']},
                minimal = minimal)
        
        return passed
    
#
if __name__ == '__main__':
    utils.newTest(TEST_ID)
    test = Test()
    test.Wrap(print_out = True)
    test.Laplacian(print_out = True)
