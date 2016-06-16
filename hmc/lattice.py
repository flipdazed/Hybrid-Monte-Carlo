import numpy as np
import traceback, sys

class Periodic_Lattice(object):
    """Creates an n-dimensional ring that joins on boundaries w/ numpy
    
    Required Inputs
        array :: np.array :: n-dim numpy array to use wrap with
    
    Only currently supports single point selections wrapped around the boundary
    """
    def __init__(self, array, spacing):
        self.get = array
        self.d = len(self.get.shape)
        self.spacing = spacing
        pass
    
    def laplacian(self, position, a_power=0):
        """lattice Laplacian for a point with a periodic boundary
        
        Required Inputs
            position :: (integer,) :: determines the position of the array
        
        Optional Inputs
            a_power  :: integer :: divide by (lattice spacing)^a_power
        
        Expectations
            position is a tuple that gives current point in the n-dim lattice
        """
        try:
            assert len(position) == self.d
        except Exception as e:
            raise ValueError("mismatch of indices...\nshape received: {}\nshape expected: {}".format(position, self.get.shape))
        position = np.asarray(position)
        
        lap = []
        position = np.asarray(position)
        for axis in xrange(self.d): # iterate through axes (lattice dimensions)
            # lattice shift operators
            # mask is a boolean mask that only contains True on the index = axis
            mask = np.in1d(np.arange(position.size), axis)
            
            # enforce periodic boundary
            plus = self.wrapIdx(position + mask)
            pos = self.wrapIdx(position)
            minus = self.wrapIdx(position - mask)
            
            lap_i = self.get[plus] - 2.*self.get[pos] + self.get[minus]
            if a_power: lap_i /= self.spacing**a_power # lattice spacing
            lap.append(lap_i)
            # next dimension
        
        return np.asarray(lap)
    
    def gradSquared(self, position, a_power=0):
        """lattice gradient^2 for a point with a periodic boundary
        
        The gradient is squared to be symmetric and avoid non differentiability
        in the continuum limit as a^2 -> 0
        -- See Feynman & Hibbs: Quantum Mechanics and Paths Integrals pg. 179
        
        Required Inputs
            position :: (integer,) :: determines the position of the array
        
        Optional Inputs
            a_power  :: integer :: divide by (lattice spacing)^a_power
        
        Expectations
            position is a tuple that gives current point in the n-dim lattice
        """
        try:
            assert len(position) == self.d
        except Exception as e:
            raise ValueError("mismatch of indices...\nshape received: {}\nshape expected: {}".format(position, self.get.shape))
        position = np.asarray(position)
        
        grad = []
        for axis in xrange(self.d): # iterate through axes (lattice dimensions)
            # lattice shift operators
            # mask is a boolean mask that only contains True on the index = axis
            mask = np.in1d(np.arange(position.size), axis)
            
            # enforce periodic boundary
            plus = self.wrapIdx(position + mask)
            pos = self.wrapIdx(position)
            minus = self.wrapIdx(position - mask)
            
            g = (self.get[plus] - self.get[pos])*(self.get[pos] - self.get[minus])
            # doesn't work if gradient changes direction between forward/backwards
            if g<0:
                g = (self.get[plus] - self.get[minus])**2 # use central difference
                if a_power: g /= (2*self.spacing)**a_power # lattice spacing
            else:
                if a_power: g /= self.spacing**a_power # lattice spacing
            
            grad.append(g)
            # next dimension
        
        return np.asarray(grad)
    
    def wrapIdx(self, index):
        """returns periodic lattice index 
        for a given iterable index
        
        Required Inputs:
            index :: iterable :: one integer for each axis
        
        This is NOT compatible with slicing
        """
        try:
            assert len(index) == len(self.get.shape)
        except AssertionError, e:
            _, _, tb = sys.exc_info()
            print '\nError in wrapIdx():'
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            print 'line {} in {}'.format(line, text)
            raise ValueError('req length: {}, length: {}'.format(len(index), len(self.get.shape)))
            sys.exit(1)
        
        mod_index = tuple(( (i%s + s)%s for i,s in zip(index, self.get.shape)))
        return mod_index
    
#
if __name__ == '__main__':
    pass