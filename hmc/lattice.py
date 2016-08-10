import numpy as np

from . import checks

class Periodic_Lattice(np.ndarray):
    """Creates an n-dimensional ring that joins on boundaries w/ numpy
    
    Required Inputs
        array :: np.array :: n-dim numpy array to use wrap with
    
    Only currently supports single point selections wrapped around the boundary
    """
    def __new__(cls, input_array, lattice_spacing=1.):
        """__new__ is called by numpy when and explicit constructor is used:
        obj = MySubClass(params) otherwise we must rely on __array_finalize
         """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        # add the new attribute to the created instance
        obj.lattice_shape = input_array.shape
        obj.lattice_dim = len(input_array.shape)
        obj.lattice_spacing = lattice_spacing
        
        # Finally, we must return the newly created object:
        return obj
    
    def __getitem__(self, index):
        index = self.latticeWrapIdx(index)
        return super(Periodic_Lattice, self).__getitem__(index)
    
    def __setitem__(self, index, item):
        index = self.latticeWrapIdx(index)
        return super(Periodic_Lattice, self).__setitem__(index, item)
    
    def __array_finalize__(self, obj):
        """ ndarray.__new__ passes __array_finalize__ the new object, 
        of our own class (self) as well as the object from which the view has been taken (obj). 
        See http://tinyurl.com/jh354s7 for more info
        """
        # ``self`` is a new object resulting from
        # ndarray.__new__(Periodic_Lattice, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. Periodic_Lattice():
        #   1. obj is None
        #       (we're in the middle of the Periodic_Lattice.__new__
        #       constructor, and self.info will be set when we return to
        #       Periodic_Lattice.__new__)
        if obj is None: return
        #   2. From view casting - e.g arr.view(Periodic_Lattice):
        #       obj is arr
        #       (type(obj) can be Periodic_Lattice)
        #   3. From new-from-template - e.g lattice[:3]
        #       type(obj) is Periodic_Lattice
        # 
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'spacing', because this
        # method sees all creation of default objects - with the
        # Periodic_Lattice.__new__ constructor, but also with
        # arr.view(Periodic_Lattice).
        #
        # These are in effect the default values from these operations
        self.lattice_shape = getattr(obj, 'lattice_shape', obj.shape)
        self.lattice_dim = getattr(obj, 'lattice_dim', len(obj.shape))
        self.lattice_spacing = getattr(obj, 'lattice_spacing', None)
        pass
    
    def latticeWrapIdx(self, index):
        """returns periodic lattice index 
        for a given iterable index
        
        Required Inputs:
            index :: iterable :: one integer for each axis
        
        This is NOT compatible with slicing
        """
        # if not hasattr(index, '__iter__'): return index         # handle integer slices
        # if len(index) != len(self.lattice_shape): return index  # must reference a scalar
        # if any(type(i) == slice for i in index): return index   # slices not supported
        try:
            if len(index) == len(self.lattice_shape):               # periodic indexing of scalars
                mod_index = tuple(( (i%s + s)%s for i,s in zip(index, self.lattice_shape)))
                return mod_index
        except:
            raise ValueError('Unexpected index: {}'.format(index))
    
#
def laplacian(lattice, position, a_power=0):
    """lattice Laplacian for a point with a periodic boundary
    
    Required Inputs
        lattice  :: Periodic_Lattice :: an overloaded numpy array
        position :: (integer,)       :: determines the position of the array
    
    Optional Inputs
        a_power  :: integer          :: divide by (lattice spacing)^a_power
    
    Expectations
        position is a tuple that gives current point in the n-dim lattice
    """
    
    # check that a lattice is input
    checks.tryAssertEqual(type(lattice), Periodic_Lattice,
         "mismatch of lattice type...\ntype received: {}\ntype expected: {}".format(
         type(lattice), Periodic_Lattice)
         )
    
    # check that the tuple recieved is the same length as the 
    # shape of the target array: Should do gradient over all dims
    # gradient should be an array of the length of degrees of freedom 
    checks.tryAssertEqual(len(position), lattice.lattice_dim,
         "mismatch of dims...\ndim received: {}\ndim expected: {}".format(
         len(position), lattice.lattice_dim)
         )
    
    # ___this is really cool___
    # see gradSquared for explanation
    plus  = position + np.identity(lattice.lattice_dim, dtype=int)
    minus = position - np.identity(lattice.lattice_dim, dtype=int)
    # repeats the position across nxn matrix
    repeated_pos = np.asarray((position,)*lattice.lattice_dim)
    lap = lattice[plus] - 2.*lattice[repeated_pos] + lattice[minus]
    # ___end cool section___
    
    # euclidean space so trivial metric
    lap = lap.sum()
    
    # multiply by approciate power of lattice spacing
    if a_power: lap = lap / lattice.lattice_spacing**a_power
    
    return lap

def gradSquared(lattice, position, a_power=0):
    """lattice gradient^2 for a point with a periodic boundary
    
    The gradient is squared to be symmetric and avoid non differentiability
    in the continuum limit as a^2 -> 0
    -- See Feynman & Hibbs: Quantum Mechanics and Paths Integrals pg. 179
    
    Required Inputs
        lattice  :: Periodic_Lattice :: an overloaded numpy array
        position :: (integer,)       :: determines the position of the array
    
    Optional Inputs
        a_power  :: integer :: divide by (lattice spacing)^a_power
    
    Expectations
        position is a tuple that gives current point in the n-dim lattice
    """
    
    # check that a lattice is input
    checks.tryAssertEqual(type(lattice), Periodic_Lattice,
         "mismatch of lattice type...\ntype received: {}\ntype expected: {}".format(
         type(lattice), Periodic_Lattice)
         )
    
    # as in laplacian()
    checks.tryAssertEqual(len(position), lattice.lattice_dim,
         "mismatch of dims...\ndim received: {}\ndim expected: {}".format(
         len(position), lattice.lattice_dim)
         )
    
    # ___this is really cool___
    # firstly:  realise that we want True on each case
    #           where the index == the given axis to shift
    # secondly: fancy indexing will index the matrix
    #           for each row in the ndarray provided
    # so this:
    # forwards = np.empty(lattice.lattice_dim)
    # all_dims = range(len(position))
    # for axis in all_dims: # iterate through axes (lattice dimensions)
    #     mask = np.in1d(all_dims, axis) # boolean mask. True on the index = axis
    #     plus = position + mask                        # lattice shift operator
    #     forwards[axis] = lattice[plus] - lattice[pos] # forwards difference
    # is equivalent to:
    plus =  position + np.identity(lattice.lattice_dim, dtype=int)
    # repeats the position across nxn matrix
    repeated_pos = np.asarray((position,)*lattice.lattice_dim)
    forwards = lattice[plus] - lattice[repeated_pos] # forwards difference
    # ___end cool section___
    
    grad = (forwards**2).sum()
    
    # lattice spacing
    if a_power: grad = grad / 1.0**a_power
    
    return grad
    
if __name__ == '__main__':
    pass