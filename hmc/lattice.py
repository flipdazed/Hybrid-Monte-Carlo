import numpy as np

from . import checks

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
        
        # check that the tuple recieved is the same length as the 
        # shape of the target array: Should do gradient over all dims
        # gradient should be an array of the length of degrees of freedom 
        checks.tryAssertEqual(len(position), self.d,
             "mismatch of indices...\nshape received: {}\nshape expected: {}".format(
             position, self.get.shape)
             )
        
        lap = np.empty(self.d)
        position = np.asarray(position) # current location
        pos = self.wrapIdx(position)    # current location (periodic)
        two_x = 2.*self.get[pos]        # current value*2
        
        # iterate through axes (lattice dimensions)
        for axis in xrange(self.d):
            # lattice shift operators
            # mask is a boolean mask that only contains True on the index = axis
            mask = np.in1d(np.arange(position.size), axis)
            
            # enforce periodic boundary
            plus = self.wrapIdx(position + mask)
            minus = self.wrapIdx(position - mask)
            
            # calculate the gradient in each dimension i
            lap[axis] = self.get[plus] - two_x + self.get[minus]
        
        # euclidean space so trivial metric
        lap = lap.sum()
        
        # multiply by approciate power of lattice spacing
        if a_power: lap /= self.spacing**a_power
        
        return lap
    
    def gradLaplacian(self, position, a_power=0):
        """gradient of the lattice Laplacian for a point 
            with a periodic boundary
        
        Required Inputs
            position :: (integer,) :: determines the position of the array
        
        Optional Inputs
            a_power  :: integer :: divide by (lattice spacing)^a_power
        
        Expectations
            position is a tuple that gives current point in the n-dim lattice
        """
        
        # check that the tuple recieved is the same length as the 
        # shape of the target array: Should do gradient over all dims
        # gradient should be an array of the length of degrees of freedom 
        checks.tryAssertEqual(len(position), self.d,
             "mismatch of indices...\nshape received: {}\nshape expected: {}".format(
             position, self.get.shape)
             )
        
        lap = self.laplacian(position, a_power=a_power)
        
        return lap
    
    def gradSquared(self, position, a_power=0, symmetric=False):
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
        
        checks.tryAssertEqual(len(position), self.d,
             "mismatch of dims...\ndim received: {}\ndim expected: {}".format(
             len(position), self.d)
             )
        
        grad = []
        position = np.asarray(position)
        forwards = np.empty(self.d)
        if symmetric: backwards = np.empty(self.d) # only assign if symmetric
        
        # iterate through axes (lattice dimensions)
        for axis in xrange(self.d):
            # lattice shift operator
            # mask is a boolean mask that only contains True on the index = axis
            mask = np.in1d(np.arange(position.size), axis)
            
            # enforce periodic boundary
            plus = self.wrapIdx(position + mask)
            pos = self.wrapIdx(position)
            
            # calculate the forwards gradient
            forwards[axis] = self.get[plus] - self.get[pos]
            
            if symmetric: # calculate the backwards gradient
                minus = self.wrapIdx(position - mask)
                backwards[axis] = self.get[pos] - self.get[minus]
            
        if symmetric:
            grad = np.dot(forwards.T, backwards)
        else:
            grad = np.dot(forwards.T, forwards)
        
        # lattice spacing
        if a_power: grad /= self.spacing**a_power
        
        return grad
    def wrapIdx(self, index):
        """returns periodic lattice index 
        for a given iterable index
        
        Required Inputs:
            index :: iterable :: one integer for each axis
        
        This is NOT compatible with slicing
        """
        
        checks.tryAssertEqual(len(index), len(self.get.shape),
             'req length: {}, length: {}'.format(len(index), len(self.get.shape))
             )
        
        mod_index = tuple(( (i%s + s)%s for i,s in zip(index, self.get.shape)))
        return mod_index
    
#
if __name__ == '__main__':
    pass