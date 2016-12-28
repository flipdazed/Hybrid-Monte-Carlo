import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def iterateSymmCartesian(ans, count,
    np.ndarray[dtype=DTYPE_t, ndim=1] sep_map,
    np.ndarray[dtype=DTYPE_t, ndim=2] op_samples,
    mean, tol, n):
    """loop the upper triangle of a cartesian product
    matching pairs with a given separation
    """
    
    for front in range(0, n): # set 1
        for back in range(0, n): # set 2
            if back >= front: continue # symsmtric opperation
            # check if front value is a new pair
            pair = np.abs(sep_map[front]-sep_map[back]) > tol
            
            if pair: # a pair has been found!
                count+= 1
                ans  += np.mean(
                    (op_samples[front] - mean)*(op_samples[back] - mean)
                )
        if front % 100 == 0: print front
    return True