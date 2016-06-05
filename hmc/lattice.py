import numpy as np
import traceback, sys

class Ring(object):
    """Creates an n-dimensional ring that joins on boundaries w/ numpy
    
    Required Inputs
        array :: np.array :: n-dim numpy array to use wrap with
    
    Only currently supports single point selections wrapped around the boundary
    """
    def __init__(self, array):
        self.get = array
        pass
    
    def wrapIdx(self, index):
        """Wraps point selection around the boundaries
        
        Required Inputs:
            index :: tuple :: one integer for each axis
        
        Expectations
            index :: same length as self.shape
        This is NOT compatible with slicing
        """
        try:
            assert len(index) == len(self.get.shape)
        except AssertionError, e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            print('Error on line {} in {}'.format(line, text))
        
        mod_index = tuple((i%s for i,s in zip(index, self.get.shape)))
        ar = self.get[mod_index]
        return ar
    
    def test(self):
        """tests the wrapping function against expected values"""
        passed = True
        ar = np.array([[ 11.,  12.,  13.,  14.],
                       [ 21.,  22.,  23.,  24.],
                       [ 31.,  32.,  33.,  34.],
                       [ 41.,  42.,  43.,  44.]])
        self.get = ar
        
        passed *= (self.wrapIdx(index=(0,0)) == 11.)
        passed *= (self.wrapIdx(index=(3,3)) == 44.)
        passed *= (self.wrapIdx(index=(4,4)) == 11.)
        passed *= (self.wrapIdx(index=(3,4)) == 41.)
        passed *= (self.wrapIdx(index=(4,3)) == 14.)
        passed *= (self.wrapIdx(index=(10,10)) == 33.)
        
        return passed
    
if __name__ == '__main__':
    test = Ring(np.empty(1))
    print 'Ring() Test:',test.test()